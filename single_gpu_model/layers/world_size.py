import torch
import torch.distributed as dist
import os
print('01')
# Define rank and world size based on the environment variables
rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
print('0011')
world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

