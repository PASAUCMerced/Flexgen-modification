# from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
#     TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
# import dataclasses
import os
import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'/home/cc/my_flexgen/single_gpu_model')
# from my_utils import get_world_size_and_world_rank
# sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/flexgen_offload')
from flexgen_utils import torch_dtype_to_np_dtype, init_weight_list 
from pytorch_backend import TorchTensor,TorchDevice, TorchDisk, TorchLink,TorchMixedDevice, DeviceType, general_copy, fix_recursive_import
DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes
# sys.path.insert(0,'/home/cc/my_flexgen/utils/')
# from cuda_mem_usage import see_memory_usage
# from cpu_mem_usage import get_memory
# dist.init_process_group(backend='nccl')

class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.name = 'SelfAttention' ####
        self.prefill = None   ####
        self.decode = False   ####
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        self.input = config.input_dim
        self.output = config.hidden_size
        
        self.num_gpus = 1 # -------------------- for simply
        
        
    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # # w_ln
            # ((h,), dtype, path + "_layer_norm.weight"),
            # # b_ln
            # ((h,), dtype, path + "_layer_norm.bias"),
        ]
        
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
        

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, prev_hidden, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head
        print('------------------************   number of head', n_head)

        donate = [False] * 14
        h, donate[0] = hidden.val, True
        print('prev_hidden', prev_hidden)
        print('prev_hidden.val', prev_hidden.val)
        print('prev_hidden.val.data', prev_hidden.val.data)
        prev_h = prev_hidden.val

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _)) = weight_read_buf.val

        if i == 0:  # prefill
            print('self attention prefill--------')
            self.prefill = True
            
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            
            h, new_k_cache, new_v_cache = self.compute.mha_wo_layernorm(prev_h, h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out,  n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            
            
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            print('self attention decode =======')
            self.prefill = False
            
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            print('self.policy.comp_cache_config ', self.policy.comp_cache_config)
            
            # h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
            #     b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
            #     k_cache, v_cache, donate, self.policy.attn_sparsity,
            #     self.policy.compress_cache, self.policy.comp_cache_config)
            
            h, new_k_cache, new_v_cache = self.compute.mha_gen_wo_layernorm(prev_h, h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h

