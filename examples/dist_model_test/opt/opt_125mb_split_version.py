"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../core/flexgen_offload/')
sys.path.insert(0,'/home/cc/my_flexgen/core/flexgen_offload')
sys.path.insert(0,'../../../dist_model/')
sys.path.insert(0,'/home/cc/my_flexgen/dist_model')

from dist_utils import initialize_distributed, get_tensor_model_parallel_group
from dist_utils import initialize_distributed_TP

from compression import CompressionConfig
from opt_config import OptConfig, get_opt_config, download_opt_weights
from device_type import DeviceType
from torch_tensor import TorchTensor, general_copy
from recursive_import import fix_recursive_import
from torch_disk import TorchDisk
from torch_link import TorchLink
from torch_device import TorchDevice
from torch_mixed_device import TorchMixedDevice
sys.path.insert(0,'/home/cc/my_flexgen/utils')   
from timers import timers

from data_types import GB, str2bool
from task  import Task
from flexgen_utils import (ExecutionEnv, ValueHolder, project_decode_latency, write_benchmark_log,read_benchmark_log)
# sys.path.insert(0,'../model')
sys.path.insert(0,'/home/cc/my_flexgen/dist_model')


from policy import Policy
from dist_optLM_model_tensor_parallel import OptLM_TP
fix_recursive_import()

import torch.distributed as dist

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    print('start create model ')
    time_m = time.time()
    print('args.rank', args.rank)
    model = OptLM_TP(opt_config, env, args.path, policy, args.local_rank)
    print('the model construction time ', time.time()-time_m)
    print('   model structure ')
    for layer in model.layers:
        print(layer.name)
        if 'Attention' in layer.name:
            print('prefill ', layer.prefill)
    print()
    


    try:
        # print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs, max_new_tokens=1, verbose=args.verbose)
        print('the useful data start from here -------------------------------------')
        print("benchmark - generate")
        timers("generate").reset()
        print('args.gen_len ', args.gen_len)
        print('input ', torch.tensor(inputs).size())
        time1 = time.time()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
        print('the model generate time ', time.time()-time1)
    finally:
        env.close_copy_threads()
    

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=2)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    # parser.add_argument("--compress-weight", action="store_true",
    #     help="Whether to compress weight.")

    # parser.add_argument("--compress-cache", action="store_true",
    #     help="Whether to compress cache.")
    # parser.add_argument("--compress-weight", type=bool,default=True)
    # parser.add_argument("--compress-cache", type=bool,default=True)
    parser.add_argument("--compress-weight", type=bool,default=False)
    parser.add_argument("--compress-cache", type=bool,default=False)

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=False)

def add_distributed_parser_arguments(parser):
    parser.add_argument('--head-ip', type=str, default=None, help='the IP address of the head node')
    parser.add_argument('--port', type=int, default=None, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=None)
    parser.add_argument('--local-rank', metavar='I', type=int, default=None)
    parser.add_argument('--world-size', metavar='N', type=int, default=None)
    parser.add_argument('--use-mpi', action='store_true', default=False,
                        help="Get distributed info from MPI")
    parser.add_argument('--comm-device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='communication through gpu nvlink or cpu memory '
                             'and socket')
    parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    parser.add_argument('--async-comm', action='store_true', default=False,
                        help="Use asynchronous communication")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # add_parser_arguments(parser)
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()
    

    if args.head_ip is not None and args.port is not None:
        print('args.head_ip ', args.head_ip)
        # print('args.port ', args.port )
        if args.use_mpi:
            # args.world_size = int(os.getenv('WORLD_SIZE','2'))
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            # print('args.world_size ', args.world_size)
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            # args.rank = int(os.getenv('RANK', '0'))
            # print('args.rank ', args.rank)
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
            # args.local_rank = int(os.getenv('local_rank', '0'))
            # print('local_rank ', args.local_rank)
            if dist.is_available():
                print("Distributed package is available!")
            else:
                print("Distributed package is not available.")
            initialize_distributed(args=args)
            print('group, ', get_tensor_model_parallel_group)
            if dist.is_initialized():
                print(f"Backend: {dist.get_backend()}")
                print(f"World Size: {dist.get_world_size()}")
                print(f"Rank: {dist.get_rank()}")
                
            else:
                print('Error: dist is not initialized()')
    #     initialize_distributed_TP(args.head_ip, args.port, args.world_size, args.rank, args.local_rank, args.comm_device)
    # else:
    #     print('not init distributed')
    #     initialize_distributed(args=args)
        
    #     args.world_size = 1
    #     args.rank = 0
    #     args.local_rank = 0


    assert len(args.percent) == 6
    
    run_flexgen(args)
