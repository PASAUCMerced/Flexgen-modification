
import torch
from torch_tensor import TorchTensor
from device_type import DeviceType
from data_types import np_dtype_to_torch_dtype, GB
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from common_utils import rms_norm
import global_config

class TorchDevice_llama:
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
        
        
    def llama_output_embed(self, inputs, w_ln, donate,
                         do_sample, temperature, lm_head, top_p):
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
