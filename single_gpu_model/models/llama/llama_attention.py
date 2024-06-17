import os
import sys
sys.path.insert(0,'/home/cc/my_flexgen/single_gpu_model')
from flexgen_utils import ExecutionEnv, init_weight_list
from policy import Policy
from pytorch_backend import general_copy, DeviceType

# sys.path.insert(0,'/home/cc/my_flexgen/examples/single_gpu_model_test/llama')
from llama_config import LlamaConfig


class LlamaAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
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

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # 5 weight files
            # w_q
            ((h, h), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((h, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((h, h), dtype, path + "self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + "self_attn.o_proj.weight"),
            # input layer norm
            ((h, ), dtype, path + "input_layernorm.weight"),
            # rotary_embed
            ((64, ), dtype, path + "self_attn.rotary_emb.inv_freq"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, w_k, w_v, w_out, input_layernorm, rotary_emb_inv_freq = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_out.smart_copy(dst1),
                input_layernorm.smart_copy(dst2),
                rotary_emb_inv_freq.smart_copy(dst2),
            ))
            
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

    def forward(
        self,
        hidden,
        cache_read_buf,
        weight_read_buf,
        attention_mask,
        cache_write_buf,
        i,
        k
    ):
        n_head = self.config.n_head

        donate = [False] * 16
        h, donate[0] = hidden.val, True

        # k is batch index
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[4]), (w_v, donate[6]), (w_out, donate[8]), (input_layernorm, donate[10]), (rotary_emb_inv_freq, donate[12])) \
                = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _),  (w_v, _), (w_out, _), (input_layernorm, _), (rotary_emb_inv_freq, _)) = weight_read_buf.val

        if i == 0:
            # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha_llama(h, mask, w_q, w_k, w_v, w_out,
                                       n_head, donate, self.policy.compress_cache, self.policy.comp_cache_config, input_layernorm, rotary_emb_inv_freq)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:
            # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen_llama(
                h, mask, w_q,
                w_k, w_v, w_out, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                input_layernorm,
                rotary_emb_inv_freq)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h