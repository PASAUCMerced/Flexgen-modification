import os
import sys
sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/flexgen_offload')
from flexgen_utils import init_weight_list

#### not finished the modification yet

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


    # load weights and bias from disk or cpu 
    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln = weight_home.val
        if k == 0:
            # dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
    
    
    def forward(self, hidden, cache_read_buf, weight_read_buf, cache_write_buf, i, k):
        
        donate = [False] * 4
        h, donate[0] = hidden.val, True

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
