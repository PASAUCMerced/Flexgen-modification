# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from enums import AttnMaskType


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    # def forward(ctx, inputs, scale):
    def forward(inputs, scale):
        import scaled_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_softmax_cuda.forward(
            inputs, scale_t[0]
        )
        # ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    # @staticmethod
    # def backward(ctx, output_grads):
    #     import scaled_softmax_cuda

    #     softmax_results, scale_t = ctx.saved_tensors

    #     input_grads = scaled_softmax_cuda.backward(
    #         output_grads, softmax_results, scale_t[0]
    #     )
    #     return input_grads, None, None
