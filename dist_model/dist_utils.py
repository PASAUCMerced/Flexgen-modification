import torch
import torch.distributed as dist
import os
import socket

_COMM_DEVICE = None
_PIPELINE_PARALLEL_PRED_GROUP = None
_PIPELINE_PARALLEL_SUCC_GROUP = None

_TENSOR_PARALLEL_GROUP = None

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None



def find_free_port():
    # Create a socket with the address family and socket type
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to a random port
    sock.bind(('localhost', 0))
    
    # Get the allocated port
    _, port = sock.getsockname()
    
    # Close the socket
    sock.close()
    
    return port


def initialize_distributed(head_ip, port, world_size, rank, local_rank,
                           comm_device):
    print(f'Initializing distributed environment at {head_ip}:{port}, '
          f'world_size={world_size}, rank={rank}, local_rank={local_rank}.')

    # Initialize distributed environment
    torch.cuda.set_device(local_rank)
    distributed_init_method = f'tcp://{head_ip}:{port}'
    global _COMM_DEVICE
    _COMM_DEVICE = comm_device
    if comm_device == 'cpu':
        backend = 'gloo'
    elif comm_device == 'gpu':
        backend = 'nccl'
    else:
        raise ValueError(f'Unknown comm_device: {comm_device}')
    dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=world_size,
                            rank=rank)

    # Create groups for pipeline parallelism
    global _PIPELINE_PARALLEL_PRED_GROUP, _PIPELINE_PARALLEL_SUCC_GROUP
    if world_size > 1:
        for pred in range(world_size):
            succ = (pred + 1) % world_size
            group = dist.new_group([pred, succ])
            if pred == rank:
                _PIPELINE_PARALLEL_PRED_GROUP = group
            if succ == rank:
                _PIPELINE_PARALLEL_SUCC_GROUP = group

    suppress_output(rank)
    print("Finished initializing distributed environment")

def get_pipeline_parallel_pred_group():
    return _PIPELINE_PARALLEL_PRED_GROUP

def get_pipeline_parallel_succ_group():
    return _PIPELINE_PARALLEL_SUCC_GROUP

def get_comm_device():
    return _COMM_DEVICE

def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', True)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs, flush=True)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_world_size():
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))
    return world_size

def initialize_distributed_TP(head_ip, port, world_size, rank, local_rank,
                           comm_device):
    
    # free_port = find_free_port()
    # print('free port ', free_port)
    # Arguments:
    #    model_parallel_size: number of GPUs used to parallelize model.
    print(f'Initializing distributed environment '
          f'world_size={world_size}, rank={rank}, local_rank={local_rank}.')

    print('local rank ', local_rank)
    torch.cuda.set_device(local_rank)
    
    # world_size = torch.distributed.get_world_size()
    


    distributed_init_method = f'tcp://{head_ip}:{free_port}'
    global _COMM_DEVICE
    _COMM_DEVICE = comm_device
    if comm_device == 'cpu':
        backend = 'gloo'
    elif comm_device == 'gpu':
        backend = 'nccl'
    else:
        raise ValueError(f'Unknown comm_device: {comm_device}')
    print('--=-=-=-=-==-=-=-====-----------   open mpi rank ', rank)

    # export MASTER_ADDR=localhost
    # export MASTER_PORT=29500
    dist.init_process_group(backend=backend,
                            init_method="env://",
                            world_size=2,
                            rank=rank)
    assert torch.distributed.is_initialized()
    #tensor_model_parallel_size (int, default = 1):
    # The number of GPUs to split individual tensors across.
    tensor_model_parallel_size = world_size
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
    
    # https://github.com/thu-coai/EVA/blob/31823a3aad848baca5d598f0b53242ff427a882c/src/mpu/initialize.py#L55
    # for i in range(world_size // model_parallel_size):
    #     ranks = range(i * model_parallel_size,
    #                   (i + 1) * model_parallel_size)
    #     group = torch.distributed.new_group(ranks)
        # if i == (rank // model_parallel_size):
        #     _TENSOR_PARALLEL_GROUP = group

    suppress_output(rank)
    print("Finished initializing -* tensor parallel *- distributed environment")
    
def get_tensor_parallel_group():
    return _TENSOR_MODEL_PARALLEL_GROUP


# def initialize_distributed(args):
#     # head_ip, port, world_size, rank, local_rank,comm_device
#     """Initialize torch.distributed."""
#     # Get local rank in case it is provided.
#     device_count = torch.cuda.device_count()
#     # print('device_count ', device_count)
    
#     global _COMM_DEVICE
#     comm_device =args.comm_device
#     _COMM_DEVICE = args.comm_device
#     if comm_device == 'cpu':
#         backend = 'gloo'
#     elif comm_device == 'gpu':
#         backend = 'nccl'
#     else:
#         raise ValueError(f'Unknown comm_device: {comm_device}')
#     local_rank = args.local_rank

#     # Get rank and world size.
#     # rank = int(os.getenv('RANK', '0'))
#     rank = args.rank
#     # world_size = int(os.getenv("WORLD_SIZE", '1'))
#     world_size = args.world_size
#     print('> initializing torch.distributed with local rank: {}, '
#           'rank: {}, world size: {}'.format(local_rank, rank, world_size))

#     # Set the device id.
#     device = rank % torch.cuda.device_count()
#     # print('----device ', device)
#     if local_rank is not None:
#         device = local_rank
#     torch.cuda.set_device(device)

#     # Call the init process.
#     init_method = 'tcp://'
#     master_ip = os.getenv('MASTER_ADDR', 'localhost')
#     master_port = os.getenv('MASTER_PORT', '29500')
#     init_method += master_ip + ':' + master_port
#     torch.distributed.init_process_group(
#         backend=backend,
#         world_size=world_size,
#         rank=rank,
#         init_method=init_method)
    
#     if torch.distributed.is_initialized():
#         print('dist init successfully')
#     else:
#         print('Error: dist is not initialized()')
        
#     tensor_model_parallel_size = world_size
#     num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
#     # Build the tensor model-parallel groups.
#     global _TENSOR_MODEL_PARALLEL_GROUP
#     assert (
#         _TENSOR_MODEL_PARALLEL_GROUP is None
#     ), 'tensor model parallel group is already initialized'
#     for i in range(num_tensor_model_parallel_groups):
#         ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
#         group = torch.distributed.new_group(ranks)
#         if rank in ranks:
#             _TENSOR_MODEL_PARALLEL_GROUP = group
            
#     suppress_output(rank)
#     print("Finished initializing -* tensor parallel *- distributed environment")
    
            

    
def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP
    
def destroy_model_parallel():
    """Set the groups to none."""
    # global _MODEL_PARALLEL_GROUP
    # _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None