import tabulate
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()


@triton.jit
def dropout_kernel(
    x_ptr, x_keep_ptr, output_ptr, p: float, n_elements: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    x_keep = tl.load(x_keep_ptr + offset, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0)

    tl.store(output_ptr + offset, output, mask=mask)


def dropout_triton(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, x_keep, output, p, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def seeded_dropout_kernel(
    x_ptr, output_ptr, p: float, n_elements: int, seed: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    random = tl.rand(seed=seed, offset=offset)
    x_keep = random > p

    x = tl.load(x_ptr + offset, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offset, output, mask=mask)


def seeded_dropout_triton(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    seeded_dropout_kernel[grid](x, output, p, n_elements, seed, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(size=(10,), device=DEVICE)
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10,), device=DEVICE) > p).to(torch.bool)
    #
    output = dropout_triton(x, x_keep=x_keep, p=p)
    print(
        tabulate.tabulate(
            [
                ["input"] + x.tolist(),
                ["keep mask"] + x_keep.tolist(),
                ["output"] + output.tolist(),
            ]
        )
    )

    output = seeded_dropout_triton(x, p=0.5, seed=123)
    output2 = seeded_dropout_triton(x, p=0.5, seed=123)
    output3 = seeded_dropout_triton(x, p=0.5, seed=512)

    print(
        tabulate.tabulate(
            [
                ["input"] + x.tolist(),
                ["output (seed = 123)"] + output.tolist(),
                ["output (seed = 123)"] + output2.tolist(),
                ["output (seed = 512)"] + output3.tolist(),
            ]
        )
    )
