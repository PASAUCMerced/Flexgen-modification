import gc
import torch
import numpy as np
import functools
from typing import  Tuple

def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem
    
def torch_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem

    

def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[S, B, H]
        indices: Tensor[K, B]
    Returns:
        Tensor[K, B, H]
    """
    S, B, H = vectors.shape
    K, B2 = indices.shape
    assert B == B2
    indices = indices.reshape(K, B, 1).expand(K, B, H)
    out = vectors.gather(dim=0, index=indices)
    return out
    
def piecewise_linear_func(xs, ys):
    """Return a function created by linear inerpolation."""
    indices = np.argsort(xs)
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # pad left and right
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    return functools.partial(piecewise_linear_func_ret_func, xs, ys)
    

def piecewise_linear_func_ret_func(xs, ys, x):
    assert x >= xs[0] and x <= xs[-1]
    return np.interp(x, xs, ys)
    
    

def sample_from_range(n, k):
    assert n >= 1

    if k == -1:
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1: return [1]
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))
        
def rms_norm(hidden_states, weight, variance_epsilon=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def precompute_freqs_cis(dim: int, end: int, inv_freq, theta= 10000.0):
    # freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # freqs = freqs.cuda()
    # inv_freq = 1.0 / (theta ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # freqs = inv_freq[: (dim // 2)]
    freqs = inv_freq
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
    
def save_file_txt(self, file_name, tensor_list):
        world_size = torch.distributed.get_world_size()
        # Define the file path
        file_path = 'files/world_size_'+str(world_size)+'/'+file_name + '_multiple_tensors.csv'

        # Open the file for writing
        with open(file_path, 'w') as file:
            # Iterate over each tensor in the list
            for i,tensor in enumerate(tensor_list):
                # Flatten the tensor
                flattened_tensor = tensor.tolist()
                file.write('tensor '+ str(i))
                for item in flattened_tensor:
                    # Write the flattened tensor as a comma-separated string
                    file.write(','.join(map(str, item)))
                    file.write('\n')
                    
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