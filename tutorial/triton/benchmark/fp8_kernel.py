"""
Usage:
python fp8_kernel.py --N 8192 --K 2048 --config-path configs/config.json
"""

import json
from argparse import ArgumentParser
from typing import Dict, List, Optional

import torch
import triton
import triton.language as tl
from sglang.srt.layers.quantization.fp8_kernel import _w8a8_block_fp8_matmul


def get_config(M: int, block_size: List[int], config_path: Optional[str] = None):
    if config_path is None:
        print(f"Using default config for {M=}")
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    key = min(configs.keys(), key=lambda x: abs(int(x) - M))
    print(f"Select {key=} for {M=}")
    return configs[key]


@triton.jit
def w8a8_block_fp8_matmul_kernel(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # scale pointers
    As_ptrs = As + offs_am * stride_As_m

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype,
    config: Dict,
):
    M, K = A.shape
    N, K = B.shape
    assert A.shape[-1] == B.shape[-1]
    assert (
        A.is_contiguous()
        and B.is_contiguous()
        and As.is_contiguous()
        and Bs.is_contiguous()
    )

    C = torch.empty((M, N), dtype=output_dtype, device="cuda")

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),
    )
    w8a8_block_fp8_matmul_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_size[0],
        block_size[1],
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        needs_masking=False,
        **config,
    )
    return C


def prepare_for_input(M, N, K, block_n, block_k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = fp8_info.min, fp8_info.max

    A = (
        torch.randn((M, K), device="cuda")
        .clamp(min=fp8_min, max=fp8_max)
        .to(torch.float8_e4m3fn)
    )
    B = (
        torch.randn((N, K), device="cuda")
        .clamp(min=fp8_min, max=fp8_max)
        .to(torch.float8_e4m3fn)
    )

    assert N % block_n == 0
    n_tiles = N // block_n
    k_tiles = K // block_k
    As = torch.randn((M, k_tiles), dtype=torch.float32, device="cuda")
    Bs = torch.randn((n_tiles, k_tiles), dtype=torch.float32, device="cuda")
    return A, B, As, Bs


def correctness_check(args):
    if args.M is None:
        Ms = [2**i for i in range(13)]
    else:
        Ms = [args.M]
    for M in Ms:
        A, B, As, Bs = prepare_for_input(M, args.N, args.K, args.block_n, args.block_k)

        # get result with default config
        block_size = [args.block_n, args.block_k]

        default_config = get_config(M, block_size, config_path=None)
        C_default = w8a8_block_fp8_matmul(
            A, B, As, Bs, block_size, output_dtype=torch.bfloat16, config=default_config
        )

        config = get_config(M, block_size, config_path=args.config_path)
        C = w8a8_block_fp8_matmul(
            A, B, As, Bs, block_size, output_dtype=torch.bfloat16, config=config
        )
        if not torch.allclose(C_default, C):
            print(f"[{M=}] Correctness Check Failed ❌")
            print(C_default)
            print(C)
            print("The maximum diff element is", torch.max(C_default - C))
        else:
            print(f"[{M=}] Correctness Check Passed ✅")


w8a8_block_fp8_matmul_after = w8a8_block_fp8_matmul


def w8a8_block_fp8_matmul_before(A, B, As, Bs, block_size, output_dtype, config):
    M, K = A.shape
    N, K = B.shape
    assert A.shape[-1] == B.shape[-1]
    assert (
        A.is_contiguous()
        and B.is_contiguous()
        and As.is_contiguous()
        and Bs.is_contiguous()
    )

    C = torch.empty((M, N), dtype=output_dtype, device="cuda")

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),
    )
    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_size[0],
        block_size[1],
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        needs_masking=False,
        **config,
    )
    return C


def get_benchmark(args):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=[2**i for i in range(13)],
            x_log=True,
            line_arg="provider",
            line_vals=["Before", "After"],
            line_names=["Before", "After"],
            ylabel="us",
            plot_name="Comparison for operator",
            args={
                "N": args.N,
                "K": args.K,
                "block_n": args.block_n,
                "block_k": args.block_k,
                "config_path": args.config_path,
            },
        )
    )
    def benchmark(M, provider, N, K, block_n, block_k, config_path):
        A, B, As, Bs = prepare_for_input(M, N, K, block_n, block_k)
        block_size = [block_n, block_k]
        config = get_config(M, block_size, config_path)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "After":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: w8a8_block_fp8_matmul_after(
                    A, B, As, Bs, block_size, output_dtype=torch.bfloat16, config=config
                ),
                quantiles=quantiles,
            )
        elif provider == "Before":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: w8a8_block_fp8_matmul_before(
                    A, B, As, Bs, block_size, output_dtype=torch.bfloat16, config=config
                ),
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"{provider=} has not been supported yet")
        return ms * 1000, min_ms * 1000, max_ms * 1000

    return benchmark


def performance_bench(args):
    benchmark = get_benchmark(args)
    benchmark.run(print_data=True)


def main(args):
    correctness_check(args)
    performance_bench(args)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path", help="The path of hyperparameter config of triton kernel"
    )
    parser.add_argument(
        "--M",
        type=int,
        help="Equal to batch size, if not set, will run all potential Ms, [2 ** i for i in range(13)]",
    )
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--block_k", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
