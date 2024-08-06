import torch
from torch_tensor import TorchTensor
from device_type import DeviceType
from data_types import np_dtype_to_torch_dtype, GB
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_tensor import general_copy
from common_utils import cpu_mem_stats, save_file_txt, rms_norm
from common_utils import precompute_freqs_cis, apply_rotary_emb
import sys
sys.path.insert(0,'/home/cc/my_flexgen/dist_model')
from dist_utils import get_tensor_model_parallel_group
from transformers.activations import ACT2FN
import global_config




class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = global_config.TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out,w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        print('mha prefill----------------')
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        
        print(' inputs.shape ',  inputs.shape)
        print('head_dim = h // n_head ', head_dim)
        # modified start--------------
        # hidden = FusedLayerNorm()
        # modified --------------end
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        
        # see_memory_usage('---------================-------------------before q, k, v \n')
        # get_memory('---------================-------------------before q, k, v \n')
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
        # see_memory_usage('---------================-------------------after mha \n')
        # get_memory('---------================-------------------after mha \n')
        return TorchTensor.create_from_torch(value, self), k, v
    
    def mha_TP(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        print('mha prefill----------------')
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        
        print(' inputs.shape ',  inputs.shape)
        print('head_dim = h // n_head ', head_dim)
        
        # hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        hidden = inputs.data
        print('hidden type', type(hidden))
        # see_memory_usage('---------================-------------------before q, k, v \n')
        # get_memory('---------================-------------------before q, k, v \n')
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
        # see_memory_usage('---------================-------------------after mha \n')
        # get_memory('---------================-------------------after mha \n')
        return TorchTensor.create_from_torch(value, self), k, v

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        print('mha_gen decode----------------')
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        
        #--------------------------- modified start
        # post_attention_layernorm = FusedLayerNorm( h, sequence_parallel=True)
        # hidden_1 = post_attention_layernorm(inputs.data,weight=w_ln.data, bias=b_ln.data )
        # print(hidden_1)
        
        # print('return ')
        # return
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data) # Add & Norm

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        # see_memory_usage('---------================-------------------after mha_gen \n')
        # get_memory('---------================-------------------after mha_gen \n')
        return TorchTensor.create_from_torch(value, self), k_new, v_new

    


    def mha_gen_TP(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        print('mha_gen decode----------------')
        
        
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        print()
        b, tgt_s, h = inputs.shape
        print('batch, tgt_sequence, hidden: '+ str(b )+" "+str(tgt_s)+' '+str(h))
        print('number of head ', n_head)
        src_s = attention_mask.shape[1]
        print('src_s ', src_s)
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        world_size = torch.distributed.get_world_size()
        print("head dimension :", head_dim)
        print('number of heads per partition: ', n_head//world_size)
        n_heads_per_partition = n_head//world_size

        print("world_size_mha_gen_TP", world_size)
        tensor_parallel_size = world_size
        rank= torch.distributed.get_rank() 
        
        # head_dim_per_partition = h // n_head//tensor_parallel_size
        # heads_per_split = n_head // tensor_parallel_size
        print('tensor_parallel_size ', tensor_parallel_size)
        print('w_q.data.shape ', w_q.data.shape)
        # tmp = w_q.data.view(n_heads_per_partition, head_dim, h)
        # print('w_q.data.view(n_heads_per_partition, head_dim, h) ', tmp.shape)
        
        q_weight_partitions = nn.ParameterList([
            nn.Parameter(w_q.data.view(tensor_parallel_size, h//tensor_parallel_size, h))
        ])
        print('len(q_weight_partitions[0] )', len(q_weight_partitions[0]))
        print('len(q_weight_partitions[0][0] )', len(q_weight_partitions[0][0]))
        print('len(q_weight_partitions[0][0][0] )', len(q_weight_partitions[0][0][0]))
        w_q_rank = q_weight_partitions[0]
        w_q_rank = w_q_rank[rank:(rank+1), : , :]
        w_q_rank = w_q_rank.squeeze(0)
        print('the shape of w_q_rank ', w_q_rank.shape)
        
        
        k_weight_partitions = nn.ParameterList([
            nn.Parameter(w_k.data.view(tensor_parallel_size, h // tensor_parallel_size, h))
        ])
        w_k_rank = k_weight_partitions[0][rank:(rank+1), :, : ].squeeze(0)
        
        
        v_weight_partitions = nn.ParameterList([
            nn.Parameter(w_v.data.view(tensor_parallel_size, h // tensor_parallel_size, h))
        ])
        w_v_rank = v_weight_partitions[0][rank:(rank+1), :, :].squeeze(0)
        
        
        out_weight_partitions = nn.ParameterList([
            nn.Parameter(w_out.data.view(tensor_parallel_size,  h, h // tensor_parallel_size))
        ])
        w_out_rank = out_weight_partitions[0][rank:(rank+1), :, :].squeeze(0).cuda(rank)
        
        
        q_bias_partitions = nn.ParameterList([
            nn.Parameter(b_q.data.view( tensor_parallel_size, h // tensor_parallel_size))
        ])
        b_q_rank = q_bias_partitions[0][rank:(rank+1), :].squeeze(0)
        print('shape of b_q_rank ', b_q_rank.shape)
        
        
        k_bias_partitions = nn.ParameterList([
            nn.Parameter(b_k.data.view( tensor_parallel_size, h // tensor_parallel_size))
        ])
        # b_k_rank = k_bias_partitions[rank]
        b_k_rank = k_bias_partitions[0][rank:(rank+1), :].squeeze(0)
        
        
        
        v_bias_partitions = nn.ParameterList([
            nn.Parameter(b_v.data.view( tensor_parallel_size, h // tensor_parallel_size))
        ])
        b_v_rank = v_bias_partitions[0][rank:(rank+1), :].squeeze(0)
        
        
        
        # out_bias_partitions = nn.ParameterList([
        #     nn.Parameter(b_out.data.view( tensor_parallel_size, h // tensor_parallel_size))
        # ])
        # b_out_rank = out_bias_partitions[0][rank:(rank+1), :].squeeze(0)
        
        
        # hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        # print('hidden size ', hidden.size())
        hidden = inputs.data

        print('hidden shape ', inputs.data.shape)
        print('input data ', inputs.data)
        # shape: (b, 1, h)
        # q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # k = F.linear(hidden, w_k.data, bias=b_k.data)
        # v = F.linear(hidden, w_v.data, bias=b_v.data)
        print('w_q_rank shape ', w_q_rank.shape)
        print('b_q_rank shape ', b_q_rank.shape)
        # F.linear(hidden, w_q_rank, bias=b_q_rank)
        # output = hidden.matmul(w_q_rank.t()) + b_q_rank
        # w_q_rank = reduce_from_tensor_model_parallel_region(output_parallel) # ?
        print('w_q_rank ', w_q_rank)
        q = (F.linear(hidden, w_q_rank, bias=b_q_rank) * scaling).cuda(rank)
        k = F.linear(hidden, w_k_rank, bias=b_k_rank).cuda(rank)
        v = F.linear(hidden, w_v_rank, bias=b_v_rank).cuda(rank)
        print('q shape after F.linear ', q.shape)
        print('k shape after F.linear ', k.shape)
        print('v shape after F.linear ', v.shape)
        # save_file_txt(' q rank '+str(rank), [q])
        # save_file_txt(' k rank '+str(rank), [k])
        # save_file_txt(' v rank '+str(rank), [v])
        # return
        # shape: (1, b, n_heads_per_partition, head_dim)
        q = q.view(tgt_s, b, n_heads_per_partition, head_dim).cuda(rank)
        k = k.view(tgt_s, b, n_heads_per_partition, head_dim).cuda(rank)
        v = v.view(tgt_s, b, n_heads_per_partition, head_dim).cuda(rank)
        print('q ', q.shape)
        print('k ', k.shape)
        print('v ', v.shape)
        print("device of q ", q.device)
        print("device of k ", k.device)
        print("device of v ", v.device)
        import copy
        rank_device = copy.deepcopy(k.device )
        #output shape :[b,n_heads_per_partition, 1, 1]
        output_size = (q.size(1), q.size(2), q.size(0), k.size(0))
        print('output_size ', output_size) # [4, 64, 1, 1]

        # [sq, b, np, hn] -> [sq, b * np, hn]
        q = q.view(output_size[2],output_size[0] * output_size[1], -1).cuda(rank)
        print('query_layer.size ', q.shape) # [1, 24, 64]

        # [sk, b, np, hn] -> [sk, b * np, hn]
        k_new = k.view(output_size[3], output_size[0] * output_size[1], -1).cuda(rank)
        print('key_layer.size ', k_new.shape) # [1, 24, 64]
        v_new = v.view(output_size[3], output_size[0] * output_size[1], -1).cuda(rank)
        print('value_layer.size ', v_new.shape) # [1, 24, 64]

        # # shape: (b * n_head, 1, head_dim)
        # q = q.permute(0, 2, 1, 3).reshape(b * n_heads_per_partition, tgt_s, head_dim)
        # # shape: (1, b * n_head, head_dim)
        # k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_heads_per_partition, head_dim)
        # shape: (1, b * n_head, head_dim)
        # v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_heads_per_partition, head_dim)
        print("device of q  new ", q.device)
        print("device of k  new", k_new.device)
        print("device of v  new", v_new.device)
        print("device of k_cache ", k_cache.device)
        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                print('enter DENSE attention------')
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_heads_per_partition, head_dim)
                    print('we do not compress cache')
                    print('q ', q.shape)
                    print('k_new shape', k_new.shape)
                    print('k_new device ', k_new.device)
                    print('v_new ', v_new.shape)
                    print('src_s ', src_s)
                    print('k_cache.data size ', len(k_cache.data))
                    print('k_cache.data size ', k_cache.data.shape)
                    k = k_cache.data[:src_s]
                    offset = output_size[0] * output_size[1]
                    k = k[:,rank*offset:(rank+1)*offset,:]
                    print('k_cache.data[:src_s] ', k.shape)
                    v = v_cache.data[:src_s]
                    print('original v_cache.data[:src_s] ', v.shape)
                    
                    v = v[:,rank*offset:(rank+1)*offset,:]
                    print('current rank partial v_cache.data[:src_s] ', v.shape)

                k[src_s - 1:src_s] = k_new # k shape : (257, 24,64)
                v[src_s - 1:src_s] = v_new # v shape : (257, 24,64)

                # shape: (b * n_heads_per_partition, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_heads_per_partition, head_dim, src_s)
                # shape: (b * n_heads_per_partition, s, head_dim)
                # v = v.permute(1, 0, 2).reshape(b * n_heads_per_partition, src_s, head_dim)
                v = v.reshape(src_s, b , n_heads_per_partition,  head_dim) #shape : (257, 4, 6, 64)
                
                print("k shape ", k.shape)
                print("v shape ", v.shape)
                print("device of k cuda ? ", k.device)
                print("device of v cuda ? ", v.device)
                if k.is_cuda:
                    print('k.is_cuda ', k.is_cuda)
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_heads_per_partition, head_dim)
                else:
                    print('k is on cpu ')
                    q = q.float().cpu()
                    k, v = k.float(), v.float() # shape (24,64,257)
                    
                    # value = self._attention_value(q, k, v, attention_mask.data,
                    #     b, src_s, tgt_s, n_head, head_dim).cuda().half()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_heads_per_partition, head_dim).to(rank_device).half()
                    
            else:  # Sparse attention
                print('enter sparse attention------')
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            print('Mixed device attention')
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim_per_partition)
        # if q,k,v are on cpu
        # shape: (b, 1, h)
        print('shape of value ', value.shape)
        print('value device ', value.device)
        print('tgt_s ', tgt_s)
        print('w_out_rank.shape ', w_out_rank.shape)
        # value = value.transpose(1, 2).view(b, tgt_s, h)

        # value shape (1,4,384) when world size = 2, 
        value = value.transpose(0, 1).view(b, tgt_s, h // tensor_parallel_size).cuda(rank)
        
        # value = value.transpose(1, 2).view(b, tgt_s, h // tensor_parallel_size)
        # print('value ', value)
        print('value.shape  before rowlinear', value.shape)
        print('w_out_rank.t() ', w_out_rank.t())
        print('w_out_rank.t().device ', w_out_rank.t().device)
        print('w_out_rank.t().shape ', w_out_rank.t().shape)
        # value = value.to(torch.float32)  # Convert 'value' to float32
        # w_out_rank = w_out_rank.to(torch.float32)  # Convert 'w_out_rank' to float32

        output_parallel = torch.matmul(value, w_out_rank.t().cuda(rank)).cuda(rank)
        print('type of output_parallel ', output_parallel.type())
        # output_parallel = F.linear(value, w_out_rank, bias=None).cuda(rank)
        print('current rank ', rank)
        print('shape of output_parallel ', output_parallel.shape )
        print('output_parallel [0][0][:8] ',output_parallel[0][0][:8])

        torch.distributed.all_reduce(output_parallel, group=get_tensor_model_parallel_group())

        print('output_parallel ', output_parallel)
        print('device of output_parallel ', output_parallel.device)
        print("shape of output_parallel ", output_parallel.shape)
        value = output_parallel + b_out.data.to(torch.cuda.current_device())
        print("shape of value after all reduce)", value.shape)

        # Layernorm on the attention output.

        
        print('inputs.data shape ', inputs.data.shape)
        print('inputs.data.device ', inputs.data.device)
        print('value shape ', value.shape)
        print('value.device ', value.device)
        
        inputs.data = inputs.data.to(torch.cuda.current_device())
        value.add_(inputs.data) # Add & Norm
        print("shape of value after add_(inputs.data)", value.shape)
        
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            print('TorchTensor.create_from_torch(value, self)')
            print('k_new device ', k_new.device)
            print('self.device ', self.dev)
            
            k_new = k_new.reshape(tgt_s, b , n_heads_per_partition,  head_dim)
            shape = k_new.shape
            new_tensor = torch.zeros([shape[0],shape[1], shape[2]*world_size,shape[3]], device=self.dev)
            print('new tensor shape ', new_tensor.shape)
            torch.distributed.all_gather(new_tensor,k_new)
            print('k_new device ', new_tensor.device)
            print('k_new shape after all gather', new_tensor.shape)
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        
        print(' create from torch')
        return TorchTensor.create_from_torch(value, self), k_new, v_new
    
    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights
    
    def _attention_weights_TP(self, q, k, mask, b, src_s, n_head):
        # the output is attention weight (or called attention score )
        # output shape: (b * n_head, 1, s) # (24, 1, 1)
        q = q.transpose(0,1) #--> (24,1,64)
        # q.shape :[ b*n_heads_per_partition, 1, head_dim] (24,1,64)
        # k.shape :[ b*n_heads_per_partition, head_dim, 257] (24, 64, 64)
        attn_weights = torch.bmm(q, k) # shape (24,1,257)
        print('attn_weights shape ', attn_weights.shape)
        print('attn_weights device', attn_weights.device)
        
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s) # (4,1,1,257)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)  # shape (4, 6, 1, 257)
        # print('attn_weights ', attn_weights)
        # print('mask.shape ', mask.shape)
        # print('mask ', mask.data) # a tensor with [true or false]
        attn_weights = torch.where(mask, attn_weights, -1e4)  # when true keep orignal weight, else set -1e4
        # print('attn_weights ', attn_weights)

        # attn_weights.shape = (4,6,1,257)
        attn_weights = F.softmax(attn_weights, dim=-1)
        print('attn_weights.shape ', attn_weights.shape)

        # attn_weights = attn_weights.view(b * n_head, 1, src_s) # shape (24, 1, 257)
        # attn_weights = F.softmax(attn_weights, dim=2)
        # print('_attention_weights shape ', attn_weights.shape)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)



    def _attention_value_TP(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        print('q, k, mask')
        print(str(q.device)+', '+str(k.device)+', '+str(mask.device))
        
        print('src_s ', src_s)
        print('n_head ', n_head)
        print('head_dim ', head_dim)
        
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        print('device of attn_weights ', attn_weights.device)
        print('attn_weights shape , ', attn_weights.shape)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        # v.shape :[ 257, b, n_heads_per_partition, head_dim] (257, 4, 6, 257)
        # context layer shape: [b, np, sq, hn]
        print('before ouput size q.shape ', q.shape)
        output_size = (v.size(1),v.size(2),q.size(0),v.size(3)) # (4,6,1,64)
        v = v.reshape(v.size(0),v.size(1)*v.size(2),v.size(3)) # (257, 24, 64)
        # change view [b * np, sq, sk]
        attention_probs = attn_weights.view(output_size[0] * output_size[1],output_size[2], -1)
        print("attention probs shape : ", attention_probs.shape) # (24, 1 257)
         # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, v.transpose(0, 1)) # (24,1,64)
        # change view [b, np, sq, hn] (4,6,1,64)
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn] =(1, 4,6,64)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]   = (1,4,384)
        new_context_layer_shape = context_layer.size()[:-2] + \
            (context_layer.size()[-2]*context_layer.size()[-1],)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer # (1,4,384)
        # # shape o return value : (b, n_head, 1, head_dim)
        # return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data) # Add & Norm
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"
        
     ## seq_len is here key states shape [-2]
    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate, token_type_embeddings):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        # mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        
        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        # token_type_ids = torch.zeros(
        #         token_ids.size(), dtype=torch.long, device=token_embed.device
        # )

        # tte = token_type_embeddings(token_type_ids).half()
        # embeddings = token_embed + tte
        embeddings = token_embed
        return TorchTensor.create_from_torch(embeddings, self)
        
        
    def llama_output_embed(self, inputs, w_ln, donate, do_sample, temperature, lm_head, top_p):
        # decompress weights
        if lm_head.device.device_type == DeviceType.COMPRESSED:
            lm_head = lm_head.device.decompress(lm_head)

        b, s, h = inputs.shape
        # hidden = inputs.data
        hidden = rms_norm(inputs.data, w_ln.data)
        # hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, lm_head.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
            # ids = sample_top_p(probs, top_p)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)
        
    def mha_llama(self, hidden_states, attention_mask, w_q, w_k, w_v, w_out, n_head, donate, compress_cache, comp_config, input_layernorm, rotary_emb_inv_freq):
        """Multi-head attention (prefill phase)."""
        # decompress weight
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        bsz,q_len,h = hidden_states.shape

        head_dim = h // n_head
        freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq.data)
        scaling = head_dim ** -0.5
        hidden = rms_norm(hidden_states.data, input_layernorm.data)
        # hidden = F.layer_norm(hidden_states.data, (h,), weight=input_layernorm.data)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)

        q = q.view(bsz, q_len, n_head, head_dim)
        k = k.view(bsz, q_len, n_head, head_dim)
        v = v.view(bsz, q_len, n_head, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[:q_len])

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(bsz * n_head, head_dim, q_len)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(bsz * n_head, q_len, head_dim)

        attn_weights = torch.bmm(q, k)
        
        idx = torch.arange(q_len, device=self.dev)
        causal_mask = (idx <= idx.view(q_len, 1)).view(1, 1, q_len, q_len)
        mask = attention_mask.data.view(bsz, 1, 1, q_len) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(bsz, n_head, q_len, q_len)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(bsz * n_head, q_len, q_len)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(bsz, n_head, q_len, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(bsz, q_len, h)
        value = F.linear(value, w_out.data)

        value.add_(hidden_states.data)

        if donate[0]: hidden_states.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v
        
        
    def mha_gen_llama(self, inputs, attention_mask, w_q, w_k, w_v,
                w_out, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, input_layernorm, rotary_emb_inv_freq):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        freq_cis = precompute_freqs_cis(head_dim, 2048 * 2, rotary_emb_inv_freq.data)
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, input_layernorm.data)
        # hidden = F.layer_norm(inputs.data, (h,), weight=input_layernorm.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis=freq_cis[src_s: src_s + tgt_s])
         # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
                        
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        
        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new
        
        
    def mlp_llama(self, inputs, gate, down, up, donate, config, post_attention_layernorm):
        if gate.device.device_type == DeviceType.COMPRESSED:
            gate = gate.device.decompress(gate)
            down = down.device.decompress(down)
            up = up.device.decompress(up)
        b, s, h = inputs.shape
        hidden_act = config.hidden_act
        act_fn = ACT2FN[hidden_act]
        src_out = rms_norm(inputs.data, post_attention_layernorm.data)
        # src_out = F.layer_norm(inputs.data, (h,), weight=post_attention_layernorm.data)
        out = F.linear(act_fn(F.linear(src_out, gate.data)) * F.linear(src_out, up.data), down.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)
