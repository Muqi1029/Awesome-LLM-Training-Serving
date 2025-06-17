# TRITON_INTERPRET=1 python load.py
import torch
import triton
import triton.language as tl


@triton.jit
def test_load(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0)
    offset = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(x_ptr + offset, mask)
    print("x val: ", x)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    BLOCK_SIZE = 8
    N = 10
    x = torch.rand(
        N,
    )
    print(f"original {x=}")
    n = triton.cdiv(N, BLOCK_SIZE)
    test_load[(n,)](x, N, BLOCK_SIZE)
