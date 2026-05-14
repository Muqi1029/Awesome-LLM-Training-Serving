from itertools import product

import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
# print(properties)
target = driver.active.get_current_target()
kernels = {}

print(f"{NUM_SM=}")
print(f"register memory={NUM_REGS * 32 / 8 / 1024}KB")
print(f"shard memory={SIZE_SMEM / 1024}KB")
print(f"warp size={WARP_SIZE}")  # always 32


def native_softmax(x: torch.Tensor):
    """x.shape = [batch_size, hidden_size]"""
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denomerator = numerator.sum(dim=1)
    ret = numerator / denomerator[:, None]
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for row_idx in tl.range(
        row_start, n_rows, step=num_programs, num_stages=num_stages
    ):
        # load one row
        cur_input_ptr = input_ptr + row_idx * input_row_stride
        offset = tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        row = tl.load(cur_input_ptr + offset, mask=mask, other=-float("inf"))

        # compute softmax logic
        row_max = tl.max(row)
        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        out = numerator / denominator

        # write to output_ptr
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + offset
        tl.store(output_ptrs, out, mask=mask)


# wrap function
def triton_softmax(x):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8

    # Number of software pipelining stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    out = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    # print(f"{n_regs=} {size_smem=}")

    # compute num_programs
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    # launch the kernel
    kernel[(num_programs, 1, 1)](
        out, x, x.stride(0), out.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages
    )
    return out


def correctness_test():
    x = torch.randn((32, 1024)).cuda()
    out = native_softmax(x)
    out_triton = triton_softmax(x)
    out_torch = torch.softmax(x, dim=-1)
    assert torch.allclose(out, out_triton), "Correctness Test Failed"
    assert torch.allclose(out, out_torch), "Correctness Test Failed"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim"],
        x_vals=list(product([2**i for i in range(8)], [2048, 4096])),
        line_arg="provider",
        line_vals=["torch", "native", "triton"],
        line_names=["PyTorch", "Native", "Triton"],
        plot_name="softmax_cmp",
        xlabel="X Shape",
        ylabel="us",
        args={},
    )
)
def benchmark(bs, dim, provider):
    x = torch.randn((bs, dim)).cuda()
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == "native":
        ms = triton.testing.do_bench(lambda: native_softmax(x))
    return ms * 1000


if __name__ == "__main__":
    correctness_test()
    benchmark.run(print_data=True, show_plots=True)
