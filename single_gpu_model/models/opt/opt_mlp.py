import os
import numpy as np 
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'/home/cc/my_flexgen/single_gpu_model')
from flexgen_utils import init_weight_list
from flexgen_utils import torch_dtype_to_np_dtype

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

# def get_choice(cur_percent, percents, choices):
#     percents = np.cumsum(percents)
#     assert np.abs(percents[-1] - 100) < 1e-5

#     for i in range(len(percents)):
#         if cur_percent < percents[i]:
#             return choices[i]
#     return choices[-1]

# def init_weight_list(weight_specs, policy, env):
#     dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
#     dev_choices = [env.disk, env.cpu, env.gpu]

#     sizes = [np.prod(spec[0]) for spec in weight_specs]
#     sizes_cumsum = np.cumsum(sizes)
#     ret = []
#     for i in range(len(weight_specs)):
#         mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
#         home = get_choice(mid_percent * 100, dev_percents, dev_choices)
#         shape, dtype, filename = weight_specs[i]

#         if len(shape) < 2:
#             pin_memory = True
#             compress = False
#         else:
#             pin_memory = policy.pin_weight
#             compress = policy.compress_weight

#         if not compress:
#             weight = home.allocate(shape, dtype, pin_memory=pin_memory)

#             if DUMMY_WEIGHT not in filename:
#                 weight.load_from_np_file(weight_specs[i][2])
#             else:
#                 weight.load_from_np(np.ones(shape, dtype))
#                 #weight.load_from_np(np.random.rand(*shape).astype(dtype))
#         else:
#             weight = home.compressed_device.allocate(
#                 shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

#             if DUMMY_WEIGHT not in filename:
#                 weight.load_from_np_file(weight_specs[i][2])
#             else:
#                 for i in range(2):
#                     x = weight.data[i]
#                     x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

#         ret.append(weight)
#     return ret



class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.name = "MLP"
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h
