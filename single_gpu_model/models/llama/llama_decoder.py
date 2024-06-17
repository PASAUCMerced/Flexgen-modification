import sys
sys.path.insert(0,'/home/cc/my_flexgen/single_gpu_model')
from flexgen_utils import ExecutionEnv, init_weight_list, ValueHolder
from policy import Policy
from pytorch_backend import general_copy, DeviceType

from llama_config import LlamaConfig
from llama_attention import LlamaAttention
from llama_mlp import LlamaMLP






class LlamaDecoderLayer:
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        self.self_attn = LlamaAttention(config=config, env=env, policy=policy, layer_id=layer_id)
        self.mlp = LlamaMLP(
            layer_id=layer_id,
            env=env,
            policy=policy,
            config=config
        )
        self.compute = self.self_attn.compute
        self.policy = policy

    def set_task(self, task):
        self.self_attn.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.self_attn.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.self_attn.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.self_attn.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.self_attn.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.self_attn.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val
        # Self Attention
        self.self_attn.forward(
            hidden=hidden,
            attention_mask=attention_mask,
            cache_read_buf=cache_read_buf,
            cache_write_buf=cache_write_buf,
            weight_read_buf=read_buf1,
            i=i,
            k=k
        )
        self.mlp.forward(hidden, cache_read_buf=None, cache_write_buf=None, weight_read_buf=read_buf2, i=i, k=k, attention_mask=attention_mask)
