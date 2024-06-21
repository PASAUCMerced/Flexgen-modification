import os
import sys
sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/flexgen_offload')
from flexgen_utils import init_weight_list

#### not finished the modification yet

class Add:
    def __init__(self, config, env, policy, layer_id):
        self.name = 'add' ####
        self.prefill = None   ####
        self.decode = False   ####
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.task = None
     
    def set_task(self, task):
        self.task = task

    # load weights from files downloaded from pretrained model(Meta)
    def init_weight(self, weight_home, path):
        pass


    # load weights and bias from disk or cpu 
    def load_weight(self, weight_home, weight_read_buf, k):
        pass

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing
    
    # self, hidden, cache_read_buf, weight_read_buf, attention_mask,cache_write_buf, i, k
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,cache_write_buf, i, k):
        
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        original_hidden = hidden.val  # Save the original hidden state in a new variable

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[2]), (b_ln, donate[3])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (b_ln, _)) = weight_read_buf.val
        
        if i == 0:  # prefill
            print('self attention prefill layernorm--------')
            self.prefill = True
            
        else:  # decoding
            print('self attention decode layernorm =======')
            self.prefill = False
        h = self.compute.layernorm(h, w_ln, b_ln, donate)
        
        hidden.val = h    
        # print('h shape', h.shape)
        # print('h', h)
