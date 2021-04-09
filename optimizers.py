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

"""The DP-FTRL optimizer."""

import torch

__all__ = ['FTRLOptimizer']


class FTRLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, momentum: float):
        """
        :param params: parameter groups
        :param momentum: if non-zero, use DP-FTRLM
        """
        defaults = dict(alpha_sum=0.0)
        self.momentum = momentum
        super(FTRLOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FTRLOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, args, closure=None):
        alpha, noise = args
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, nz in zip(group['params'], noise):
                if p.grad is None:
                    continue
                d_p = p.grad

                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['grad_sum'] = torch.zeros_like(d_p, memory_format=torch.preserve_format)
                    param_state['model_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    param_state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    param_state['model_sum'].add_(p)  # just record the initial model

                gs, ms = param_state['grad_sum'], param_state['model_sum']
                if self.momentum == 0:
                    gs.add_(d_p)
                    p.copy_(ms + (-gs - nz) / alpha)
                else:
                    gs.add_(d_p)
                    param_state['momentum'].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms - param_state['momentum'] / alpha)
        return loss
