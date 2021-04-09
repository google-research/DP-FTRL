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

"""DP-FTRL training, based on paper
"Practical and Private (Deep) Learning without Sampling or Shuffling"
https://arxiv.org/abs/2103.00039.
"""

from absl import app
from absl import flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import PrivacyEngine

from nn import get_nn
from data import get_data
from optimizers import FTRLOptimizer
from privacy import compute_epsilon_tree
from ftrl_noise import CummuNoise
import utils
from utils import EasyDict


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], '')

flags.DEFINE_boolean('dp_ftrl', True, 'If True, train with DP-FTRL. If False, train with vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 4.0, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_boolean('restart', False, 'If True, restart the tree after each epoch.')
flags.DEFINE_float('momentum', 0, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 250, 'Batch size.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs.')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')

flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')


def main(argv):
    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data and the privacy delta
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    delta = {'mnist': 1e-5, 'cifar10': 1e-5, 'emnist_merge': 1e-6}[FLAGS.data]
    print('Training set size', trainset.image.shape)

    # Hyperparameters for training.
    epochs = FLAGS.epochs
    batch = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain
    num_batches = ntrain // batch
    noise_multiplier = FLAGS.noise_multiplier if FLAGS.dp_ftrl else -1
    clip = FLAGS.l2_norm_clip if FLAGS.dp_ftrl else -1
    lr = FLAGS.learning_rate

    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    assert report_nimg % batch == 0

    # Get the name of the output directory.
    log_dir = os.path.join(FLAGS.dir, FLAGS.data,
                           utils.get_fn(EasyDict(batch=batch),
                                        EasyDict(dpsgd=FLAGS.dp_ftrl, restart=FLAGS.restart, noise=noise_multiplier, clip=clip, mb=1),
                                        [EasyDict({'lr': lr}),
                                         EasyDict(m=FLAGS.momentum if FLAGS.momentum > 0 else None),
                                         EasyDict(sd=FLAGS.run)]
                                        )
                           )
    print('Model dir', log_dir)

    # Function to output batches of data
    def data_stream():
        while True:
            perm = np.random.permutation(ntrain)
            for i in range(num_batches):
                batch_idx = perm[i * batch:(i + 1) * batch]
                yield trainset.image[batch_idx], trainset.label[batch_idx]

    # Function to conduct training for one epoch
    def train_loop(model, device, optimizer, cumm_noise, epoch, writer):
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        loop = trange(0, num_batches * batch, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, epochs))
        step = epoch * num_batches
        for it in loop:
            step += 1
            data, target = next(data_stream())
            data = torch.Tensor(data).to(device)
            target = torch.LongTensor(target).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step((lr, cumm_noise()))
            losses.append(loss.item())

            if (step * batch) % report_nimg == 0:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
                model.train()
                print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))

        eps = compute_epsilon_tree(FLAGS.restart, epoch + 1, num_batches, noise_multiplier, delta)
        writer.add_scalar('privacy/eps', eps, epoch + 1)
        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f Privacy %.4f' % (epoch + 1, np.mean(losses), eps))

    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            for i, dataset in enumerate([trainset, testset]):
                for it in trange(0, dataset.image.shape[0], b, leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data, target = torch.Tensor(data).to(device), torch.LongTensor(target).to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    accs[i] += pred.eq(target.view_as(pred)).sum().item()
                accs[i] /= dataset.image.shape[0]
        return accs

    # Get model for different dataset
    device = torch.device('cuda')
    model = get_nn({'mnist': 'small_nn',
                    'emnist_merge': 'small_nn',
                    'cifar10': 'vgg128'}[FLAGS.data],
                   nclass=nclass).to(device)

    # Set the (DP-)FTRL optimizer. For DP-FTRL, we
    # 1) use the opacus library to conduct gradient clipping without adding noise
    # (so we set noise_multiplier=0). Also we set alphas=[] as we don't need its
    # privacy analysis.
    # 2) use the CummuNoise module to generate the noise using the tree aggregation
    # protocol. The noise will be passed to the FTRL optimizer.
    optimizer = FTRLOptimizer(model.parameters(), momentum=FLAGS.momentum)
    if FLAGS.dp_ftrl:
        privacy_engine = PrivacyEngine(model, batch_size=batch, sample_size=ntrain, alphas=[], noise_multiplier=0, max_grad_norm=clip)
        privacy_engine.attach(optimizer)
    shapes = [p.shape for p in model.parameters()]
    cumm_noise = CummuNoise(noise_multiplier * clip / batch, shapes, device)

    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    for epoch in range(epochs):
        if FLAGS.restart:  # if restarting the tree aggregation, we'll setup a new optimizer and noise module
            optimizer = FTRLOptimizer(model.parameters(), momentum=FLAGS.momentum)
            if FLAGS.dp_ftrl:
                privacy_engine.detach()
                privacy_engine.attach(optimizer)
            cumm_noise = CummuNoise(noise_multiplier * clip / batch, shapes, device)
        train_loop(model, device, optimizer, cumm_noise, epoch, writer)
    writer.close()


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
