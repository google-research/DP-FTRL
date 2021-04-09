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

"""Privacy computation for DP-FTRL."""

import math
import numpy as np


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


def compute_epsilon_tree(restart, epochs, num_batches, noise, delta, verbose=True):
    """
    Compute epsilon value for DP-FTRL.
    :param restart: whether to restart the tree after each epoch
    :param epochs: number of epochs
    :param num_batches: number of batches per epoch
    :param noise: noise multiplier for each step
    :param delta: target DP delta
    :param verbose: whether to print message
    :return: the DP epsilon for DP-FTRL
    """
    if noise < 1e-20:
        return float('inf')
    if not restart:
        num_batches *= epochs
    log_num_batches = np.ceil(np.log2(num_batches))
    eps = convert_gaussian_renyi_to_dp(noise / math.sqrt(log_num_batches * epochs),
                                       delta, verbose)
    return eps

