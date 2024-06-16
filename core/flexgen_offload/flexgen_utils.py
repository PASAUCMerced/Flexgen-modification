
import dataclasses

from attr.setters import frozen


import os
from typing import  Any

import numpy as np
import torch
from data_types import GB, torch_dtype_to_np_dtype

import torch.distributed as dist

# dist.init_process_group(backend='gloo')
# dist.init_process_group(backend='nccl')
# world_size = dist.get_world_size()
# world_rank = dist.get_rank()



DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes









@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        from my_flexgen.flexgen_offload.torch_mixed_device import TorchDevice, TorchDisk, TorchMixedDevice
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results."""
    prefill_latency: float
    prefill_throughput: float
    decode_latency: float
    decode_throughput: float
    total_latency: float
    total_throughput: float












class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None
        
    # def get_length(self):
    #     """
    #     Get the length of the 'val' attribute if it's a sequence or None if it's not.
    #     """
    #     if self.val is None:
    #         return 0
    #     elif isinstance(self.val, (list, tuple, str)):
    #         return len(self.val)
    #     else:
    #         raise ValueError("ValueHolder 'val' is not a sequence (list, tuple, or string).")







def run_cmd(cmd):
    print(cmd)
    os.system(cmd)




def project_decode_latency(costs, prompt_len, gen_len):
    decode_costs = costs[1:]

    if gen_len / prompt_len < 0.1:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))
    else:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))

        #assert len(decode_costs) >= 4
        #warmup = 2
        #xs = np.arange(warmup, len(decode_costs))
        #ys = np.asarray(decode_costs[warmup:])
        #curve = np.poly1d(np.polyfit(xs, ys, deg=1))
        #ys_pred = [curve(x) for x in range(gen_len-1)]
        #decode_latency = sum(ys_pred)

        #print([round(x, 4) for x in decode_costs])
        #print([round(x, 4) for x in ys_pred])

    return decode_latency


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):

    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"projected: {projected}\n"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str


def read_benchmark_log(filename):
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]
        throughput = b[b.index(":") + 1:b.index(" to")]
        return float(latency), float(throughput)

    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )
    
    


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        print('*********-------=-=-=--mid_percent ', mid_percent)
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        print('home device is ', home)
        shape, dtype, filename = weight_specs[i]
        print("weight_specs[i] ", weight_specs[i])

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
                print('weight shape ', weight.shape)
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
        # print('ret ', ret)
        # print('ret[0] ', ret[0].shape)
        # if len(ret)>1:
        #     print('ret[1] ', ret[1].shape)
    return ret


def init_weight_list_tensor_parallel(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        print('*********-------=-=-=--mid_percent ', mid_percent)
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        print('home device is ', home)
        shape, dtype, filename = weight_specs[i]
        print("weight_specs[i] ", weight_specs[i])

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
                print('load_from_np_file,  weight shape ', weight.shape)
                # dist.init_process_group(backend='nccl')
                world_size = dist.get_world_size()
                world_rank = dist.get_rank()
                
                
                output_size_per_partition = divide(output_size, world_size)

                weight_list = torch.split(weight, output_size_per_partition, dim=1)
                
                
                my_weight_list = weight_list[rank::world_size]
            else:
                print('DUMMY weights ')
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
        # print('ret ', ret)
        # print('ret[0] ', ret[0].shape)
        # if len(ret)>1:
        #     print('ret[1] ', ret[1].shape)
    return ret
