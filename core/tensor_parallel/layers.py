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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from initialize import get_tensor_model_parallel_rank
from initialize import get_tensor_model_parallel_world_size
from initialize import get_tensor_model_parallel_group
from mappings import copy_to_tensor_model_parallel_region
from mappings import gather_from_tensor_model_parallel_region
from mappings import gather_from_sequence_parallel_region
from mappings import reduce_from_tensor_model_parallel_region
from mappings import scatter_to_tensor_model_parallel_region
from mappings import reduce_scatter_to_sequence_parallel_region

from mpu_random import get_cuda_rng_tracker
from utils import divide
from utils import split_tensor_along_last_dim
from utils import VocabUtility
from global_vars import get_args, get_global_memory_buffer

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}



class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            if args.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            if args.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                args.sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, bias, self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce, self.sequence_parallel)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias






# import torch
# import torch.nn as nn

# class ColumnParallelLinear(nn.Module):
#     def __init__(self, in_features, out_features, tensor_parallel_size):
#         super(ColumnParallelLinear, self).__init__()
#         self.tensor_parallel_size = tensor_parallel_size
#         self.head_dim = in_features // tensor_parallel_size
        
#         # Assuming weights and biases are predefined tensors
#         self.weight = torch.randn(in_features, out_features)
#         self.bias = torch.randn(out_features)
        
#         # Create weight partitions for tensor parallelism
#         self.weight_partitions = nn.ParameterList([
#             nn.Parameter(self.weight[:, i * self.head_dim: (i + 1) * self.head_dim])
#             for i in range(tensor_parallel_size)
#         ])
        
#         # Create bias partitions for tensor parallelism
#         self.bias_partitions = nn.ParameterList([
#             nn.Parameter(self.bias[i * self.head_dim: (i + 1) * self.head_dim])
#             for i in range(tensor_parallel_size)
#         ])
    
#     def forward(self, x, split_idx):
#         weight = self.weight_partitions[split_idx]
#         bias = self.bias_partitions[split_idx]
#         return torch.matmul(x, weight) + bias

# Self-Attention Layer with Tensor Parallelism Size 2
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, tensor_parallel_size):
        super(SelfAttention, self).__init__()
        self.tensor_parallel_size = tensor_parallel_size
        self.head_dim = embed_dim // tensor_parallel_size
        
        # ColumnParallelLinear for query, key, value, and output
        self.q_linear = ColumnParallelLinear(embed_dim, embed_dim, tensor_parallel_size)
        self.k_linear = ColumnParallelLinear(embed_dim, embed_dim, tensor_parallel_size)
        self.v_linear = ColumnParallelLinear(embed_dim, embed_dim, tensor_parallel_size)
        self.out_linear = ColumnParallelLinear(embed_dim, embed_dim, tensor_parallel_size)
    
    def forward(self, x, split_idx):
        q = self.q_linear(x, split_idx)
        k = self.k_linear(x, split_idx)
        v = self.v_linear(x, split_idx)
        
#         # Perform attention calculation, weighted sum, etc. (not shown)
        
#         output = self.out_linear(weighted_sum, split_idx)
#         return output

# # Example usage
# embed_dim = 64
# tensor_parallel_size = 2

# self_attention = SelfAttention(embed_dim, tensor_parallel_size)
# input_data = torch.randn(32, embed_dim)  # Batch size: 32, Embedding dimension: 64
# split_idx = 0  # Use GPU 0 for this example

# output = self_attention(input_data, split_idx)
# print("Output Shape:", output.shape)

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 
                 params_dtype=torch.float32,
                 
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
            setattr(self.bias, 'sequence_parallel', sequence_parallel_enabled)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )

        # All-reduce across all the partitions.
        
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = output_
        output_bias = self.bias
        
        return output, output_bias