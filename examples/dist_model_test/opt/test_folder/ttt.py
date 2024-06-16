import os
import torch
import torch.distributed as dist
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # Set the GPU device
    torch.cuda.set_device(args.local_rank)

    # Initialize the distributed environment
    dist.init_process_group(
        backend='nccl',  # Use 'nccl' backend for GPU communication
        init_method='tcp://localhost:FREE_PORT',  # Use 'tcp' method for communication
        rank=args.local_rank,
        world_size=2  # Total number of GPUs you want to use
    )

    # Your distributed training code here

if __name__ == "__main__":
    main()
