# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Privacy computation for DP-FTRL with given data order."""

from absl import app
import numpy as np
from typing import List
from collections import Counter
import json


def convert_gaussian_renyi_to_dp(sigma, delta, verbose=True):
    """
    Convert from RDP to DP for a Gaussian mechanism.
    :param sigma: the algorithm guarantees (alpha, alpha/(2*sigma^2))-RDP
    :param delta: target DP delta
    :param verbose: whether to print message
    :return: the DP epsilon
    """
    alphas = np.arange(1, 200, 0.1)[1:]
    epss = alphas / 2 / sigma**2 - (np.log(delta*(alphas - 1)) - alphas * np.log(1 - 1/alphas)) / (alphas - 1)
    idx = np.nanargmin(epss)
    if verbose and idx == len(alphas) - 1:
        print('The best alpha is the last one. Consider increasing the range of alpha.')
    eps = epss[idx]
    return eps


def get_total_sensitivity_sq_given_order(order):
    """
    Get the squared sensitivity for a given order of batches.
    Can be viewed as a general case for get_total_sensitivity_sq_same_order.
    This function is not used in the privacy computation, as we operated in the case where the
    data order is the same for every epoch.

    :param order: a list representing the order of the batches, e.g. [0,1,2,1] means we use batch indexed with 0,1,2,1.
                  -1 indicates virtual step.
    :return: squared sensitivity, squared_sensitivity with respect to all every batch
    """
    # get first layer as a list of counters
    layer = [Counter({node: 1}) for node in order]

    # sensitivity_sq[i] will record the total sensitivity wrt batch i
    sensitivity_sq_all = [0] * (max(order) + 1)

    # update sensitivity_sq with a given layer
    def update_sensitivity_sq(current_layer):
        for node in current_layer:
            for ss in node:
                if ss != -1:
                    sensitivity_sq_all[ss] += node[ss] ** 2

    update_sensitivity_sq(layer)  # get sensitivity for the first layer
    while len(layer) > 1:
        layer_new = []  # merge every two consecutive nodes to get the next layer
        length = len(layer)
        for i in range(0, length, 2):
            if i + 1 < length:
                layer_new.append(layer[i] + layer[i + 1])
        layer = layer_new
        update_sensitivity_sq(layer)
    return max(sensitivity_sq_all), sensitivity_sq_all


def get_total_sensitivity_sq_same_order(steps_per_epoch, epochs, extra_steps, mem_fn=None):
    """
    Get the squared sensitivty for a tree where we fix the order of batches for all epochs.

    :param steps_per_epoch: number of steps per epoch
    :param epochs: number of epochs in the tree
    :param extra_steps: number of virtual steps
    :param mem_fn: if set, will write result to the file
    :return: squared sensitivity, squared sensivity assuming no virtual steps,
             squared sensitivity with respect to every batch

    e.g. steps_per_epochs = 3 and epochs = 2, extra_steps = 2 means we have three batches b1, b2, b3,
    and train w/ [b1, b2, b3, b1, b2, b3, +, +] where + means the extra steps.
    We will enumerate through all nodes layer by layer in list "layer", and  compute the sensitivty
    with respect to every node in "sensitivity_sq".
    """
    # to record the result to save computation
    mem = json.load(open(mem_fn)) if mem_fn else {}
    key = f'{steps_per_epoch},{epochs},{extra_steps}'
    key_no_extra = f'{steps_per_epoch},{epochs},{0}'
    if key in mem and key_no_extra in mem:
        return mem[key], mem[key_no_extra], None

    # get first layer as a list of counters, the keys are batches (indexed with non-negative numbers), counts are
    # number of times the batch appears in the node
    layer = []
    for _ in range(epochs):
        layer += [Counter({ss: 1}) for ss in range(steps_per_epoch)]
    layer += [Counter({-1: 1}) for _ in range(extra_steps)]  # extra steps denoted as -1

    # sensitivity_sq[i] will record the total sensitivity wrt batch i
    sensitivity_sq_all = [0] * steps_per_epoch
    sensitivity_sq_all_no_extra = [0] * steps_per_epoch  # will also compute sensitivity without extra

    # update sensitivity_sq with a given layer
    def update_sensitivity_sq(current_layer):
        for node in current_layer:
            has_extra = -1 in node
            for ss in node:
                if ss != -1:
                    sensitivity_sq_all[ss] += node[ss] ** 2
                    if not has_extra:
                        sensitivity_sq_all_no_extra[ss] += node[ss] ** 2

    update_sensitivity_sq(layer)  # get sensitivity for the first layer
    while len(layer) > 1:
        layer_new = []  # merge every two consecutive nodes to get the next layer
        length = len(layer)
        for i in range(0, length, 2):
            if i + 1 < length:
                layer_new.append(layer[i] + layer[i + 1])
        del layer
        layer = layer_new
        update_sensitivity_sq(layer)

    # save to file
    if mem_fn:
        mem[key] = max(sensitivity_sq_all)
        mem[key_no_extra] = max(sensitivity_sq_all_no_extra)
        with open(mem_fn, 'w') as f:
            json.dump(mem, f, indent=4)
    return max(sensitivity_sq_all), max(sensitivity_sq_all_no_extra), sensitivity_sq_all


def compute_epsilon_tree_restart_rdp_same_order_extra(num_batches: int, epochs_between_restarts: List[int],
                                                      noise: float, tree_completion: bool = True,
                                                      mem_fn: str = None):
    """
    Compute the effective noise for DP-FTRL.

    :param num_batches: number of batches per epoch
    :param epochs_between_restarts: number of epochs between each restart, e.g. [2, 1] means epoch1, epoch2, restart, epoch3
    :param noise: noise multiplier for each step
    :param tree_completion: if true, use the tree completion trick which adds virtual steps to complete the binary tree
    :param mem_fn: if set, will write result to the file
    :return: the effective noise for DP-FTRL
    """
    if noise < 1e-20:
        return float('inf')

    mem = {}  # to record result to avoid computing the same setting twice
    sensitivity_sq = 0  # total sensitivity^2, which is the sum over all "intervals" between each restarting
    for i, epochs in enumerate(epochs_between_restarts):
        if epochs == 0:
            continue
        if tree_completion and i < len(epochs_between_restarts) - 1:
            # compute number of virtual steps
            extra_steps = 2 ** (num_batches * epochs - 1).bit_length() - num_batches * epochs
        else:
            extra_steps = 0
        key = (num_batches, epochs, extra_steps)
        mem[key] = mem.get(key,
                           get_total_sensitivity_sq_same_order(num_batches, epochs, extra_steps, mem_fn)[0])
        sensitivity_sq += mem[key]
    effective_sigma = noise / np.sqrt(sensitivity_sq)
    return effective_sigma


def compute_epsilon_tree(num_batches: int, epochs_between_restarts: List[int], noise: float, delta: float,
                         tree_completion: bool,
                         verbose=True, mem_fn=None):
    """
    Compute epsilon value for DP-FTRL.

    :param num_batches: number of batches per epoch
    :param epochs_between_restarts: number of epochs between each restart, e.g. [2, 1] means epoch1, epoch2, restart, epoch3
    :param noise: noise multiplier for each step
    :param delta: target DP delta
    :param tree_completion: if true, use the tree completion trick which adds virtual steps to complete the binary tree
    :param verbose: whether to print message
    :param mem_fn: if set, will write result to the file
    :return: the DP epsilon for DP-FTRL
    """

    if noise < 1e-20:
        return float('inf')

    effective_sigma = compute_epsilon_tree_restart_rdp_same_order_extra(num_batches, epochs_between_restarts, noise,
                                                                        tree_completion, mem_fn)
    eps = convert_gaussian_renyi_to_dp(effective_sigma, delta, verbose)
    return eps


def main(_):
    # An example for CIFAR-10 (50000 samples) with batch=500, restarting every 20 epochs for 100 epochs in total,
    # noise=46.3, using the tree completion trick.
    n = 50000
    delta = 1e-5
    batch = 500
    epochs = 100
    restart_every = 20
    noise = 46.3
    tree_completion = True

    num_batches = n // batch
    epochs_between_restarts = [restart_every] * (epochs // restart_every)

    eps = compute_epsilon_tree(num_batches, epochs_between_restarts, noise, delta, tree_completion)

    print(f'n={n}, batch={batch}, epochs={epochs} with restarting every {restart_every} epochs',
          f'noise={noise}, tree_completion={tree_completion}',
          f'gives ({eps:.2f}, {delta})-DP')


if __name__ == '__main__':
    app.run(main)

