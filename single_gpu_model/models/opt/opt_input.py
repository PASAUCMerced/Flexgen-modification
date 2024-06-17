import os
import numpy as np
import sys
sys.path.insert(0,'/home/cc/my_flexgen/single_gpu_model')

from flexgen_utils import init_weight_list

class InputEmbed:
    def __init__(self, config, env, policy):
        self.name = "InputEmbed"
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        if self.config.max_seq_len >0 : # opt
            v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
                self.config.max_seq_len, self.config.dtype)
            path = os.path.join(path, "")
            weight_specs = [
                # w_token
                ((v, h), dtype, path + "decoder.embed_tokens.weight"),
                # w_pos
                ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
            ]

        else: ## llama
            v, h, dtype = (self.config.vocab_size, self.config.input_dim, self.config.dtype)
            path = os.path.join(path, "")
            weight_specs = [
                ((v, h), dtype, path + "embed_tokens.weight"),
                ]
            
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h

