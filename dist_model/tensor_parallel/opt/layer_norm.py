import os
import sys
sys.path.insert(0,'../flexgen_offload')
sys.path.insert(0,'/home/cc/new_flexgen/flexgen_offload')
from flexgen_utils import init_weight_list
from device_type import DeviceType
# from torch_tensor import TorchTensor, general_copy
# from recursive_import import fix_recursive_import
# from torch_disk import TorchDisk
# from torch_link import TorchLink
# from torch_device import TorchDevice
# from torch_mixed_device import TorchMixedDevice



class Layer_norm:
    def __init__(self, config, env, policy, layer_id):
        self.name = 'layer_norm' ####
        self.prefill = None   ####
        self.decode = False   ####
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.task = None
     
        
    # load weights from files downloaded from pretrained model(Meta)
    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        # path = os.path.join(path, "")
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    def set_task(self, task):
        self.task = task
        
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
        
    # load weights and bias from disk or cpu 
    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln = weight_home.val
        if k == 0:
            # dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
    
    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing
    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing
    
    def forward(self, hidden, cache_read_buf, weight_read_buf, cache_write_buf, attention_mask, i, k):
        
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        # mask, donate[1] = attention_mask.val.smart_copy(self.compute)


        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[2]), (b_ln, donate[3])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (b_ln, _)) = weight_read_buf.val
        
        if i == 0:  # prefill
            print('self attention prefill--------')
            self.prefill = True
            
        else:  # decoding
            print('self attention decode =======')
            self.prefill = False
        
        hidden.val = h    
