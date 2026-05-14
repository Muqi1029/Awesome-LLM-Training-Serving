import torch
import triton
import triton.language as tl


@triton.jit
def row_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    a_ptr: (M, K)
    b_ptr: (K, N)
    c_ptr: (M, N)
    grid: num_pid_m * num_pid_n
    """
    pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offset_a_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_a_row = offset_a_row < M

    offset_b_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_b_col = offset_b_col < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        # load a
        offset_a_col = k + tl.arange(0, BLOCK_SIZE_K)
        mask_a_col = offset_a_col < K
        offset_a = offset_a_row[:, None] * stride_am + offset_a_col[None, :] * stride_ak
        mask_a = mask_a_row[:, None] & mask_a_col[None, :]
        a = tl.load(a_ptr + offset_a, mask=mask_a, other=0.0)

        # load b
        offset_b_row = k + tl.arange(0, BLOCK_SIZE_K)
        mask_b_row = offset_b_row < K
        offset_b = offset_b_row[:, None] * stride_bk + offset_b_col[None, :] * stride_bn
        mask_b = mask_b_row[:, None] & mask_b_col[None, :]
        b = tl.load(b_ptr + offset_b, mask=mask_b, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=False)

    # store acc
    offset_c_row = offset_a_row
    mask_c_row = offset_c_row < M
    offset_c_col = offset_b_col
    mask_c_col = offset_c_col < N
    offset_c = offset_c_row[:, None] * stride_cm + offset_c_col[None, :] * stride_cn
    mask_c = mask_c_row[:, None] & mask_c_col[None, :]
    tl.store(c_ptr + offset_c, acc, mask=mask_c)


def row_matmul(A: torch.Tensor, B: torch.Tensor):
    # check
    M, K = A.shape
    K_b, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype).to(A.device)
    assert K_b == K
    assert C.shape == (M, N)
    assert A.is_contiguous() and B.is_contiguous()

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    row_matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(4, 8)],
        line_arg="provider",
        line_vals=["row_matmul", "col_matmul", "torch"],
        line_names=["row_matmul", "col_matmul", "torch"],
        xlabel="N",
        ylabel="GB/s",
        args={},
    )
)
def benchmark(N, provider):
    pass


if __name__ == "__main__":
    M, N, K = 512, 1024, 258
    A = torch.randn((M, K)).cuda()
    B = torch.randn((K, N)).cuda()
    C_row_triton = row_matmul(A, B)
    C_torch = torch.matmul(A, B)

    max_diff = torch.max(torch.abs(C_row_triton - C_torch))
    print(f"Max difference: {max_diff:.6e}")
    if torch.allclose(C_row_triton, C_torch):
        print("✅ Correctness check passed!")
    else:
        print("❌ Correctness check failed!")
