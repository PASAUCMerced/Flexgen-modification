from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
#from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from flexgen.dist_flex_opt import DistOptLM, OptLM

class DistOptLMBackend(ModuleBackend, DistOptLM):
    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.max_chunk_size_bytes = max_chunk_size_bytes

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.generation_pool = PrioritizedTaskPool(
            self.generation_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )


    ## connect for loop in dist_flex_opt & send receive
    def generation_step(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    # timers(timer_name).start()
                    for k in range(self.num_gpu_batches):
                        self.update_attention_mask(b, t, i, k)

                    # if self.num_pipeline_stages > 1:
                    #     self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):
                        for k in range(self.num_gpu_batches):
                            self.load_weight(b, t, i, j, k)
                        self.sync()

                        for k in range(self.num_gpu_batches):
                            self.load_cache(t, i, j, k)
                            self.load_hidden(b, t, i, j, k)
                            self.sync()
                            self.compute_layer(t, i, j, k)
                            self.sync()
                            self.store_hidden(b, t, i, j, k)
                            self.store_cache(t, i, j, k)
                            self.sync()