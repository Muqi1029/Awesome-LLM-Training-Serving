import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()


@triton.jit
def layernorm_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    row_stride: int,
    N: int,  # col: hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    input_ptr: shape [batch_size, hidden_dim]
    output_ptr: shape [batch_size, hidden_dim]
    mean_ptr: shape [batch_size]
    rstd_ptr: shape [batch_size]

    weight_ptr: shape [batch_size, hidden_dim]
    bias_ptr: shape [batch_size, hidden_dim]
    """
    row_idx = tl.program_id(0)

    offset_start_idx = row_idx * row_stride

    # Step 1: compute mean
    _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for idx in tl.range(0, N, BLOCK_SIZE):
        offset = offset_start_idx + idx + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        _mean += tl.load(input_ptr + offset, mask=mask, other=0.0)
    mean = tl.sum(_mean, axis=0)

    # Step 2: compute rstd
    _var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for idx in tl.range(0, N, BLOCK_SIZE):
        offset = offset_start_idx + idx + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(input_ptr + offset, mask=mask, other=0.0)
        _var += (x - mean) * (x - mean)
    var = tl.sum(_var, axis=0)
    rstd = 1 / tl.sqrt(var)

    # Step 3: store mean rstd
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

    # Step 4: Apply weight and bias (LayerNorm)
    for idx in tl.range(0, N, BLOCK_SIZE):
        offset = offset_start_idx + idx + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(input_ptr + offset, mask=mask)
        w = tl.load(weight_ptr + offset, mask=mask)
        b = tl.load(bias_ptr + offset, mask=mask)
        output = (x - mean) * rstd * w + b
        tl.store(output_ptr + offset, output, mask=mask)


def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
):
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
    output = torch.empty_like(x)

    N = x.shape[-1]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    layernorm_kernel[grid](
        x, output, mean, rstd, weight, bias, x.stride(0), N, BLOCK_SIZE=128
    )
    return output
