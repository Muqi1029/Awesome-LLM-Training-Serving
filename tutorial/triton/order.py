import torch
import triton
import triton.language as tl


@triton.jit
def trace_order_kernel(order_ptr, counter_ptr, num_progs):
    pid = tl.program_id(axis=0)
    # 原子加 1，返回加之前的旧值，代表“我是第几个被启动的”
    arrival_order = tl.atomic_add(counter_ptr, 1)
    # 把序号存入对应的位置
    tl.store(order_ptr + pid, arrival_order)


def test_order():
    N = 1024
    order = torch.zeros(N, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    trace_order_kernel[(N,)](order, counter, N)

    print(order[:20])  # 查看前20个 Block 的启动顺序
    # 如果输出接近 [0, 1, 2, 3, 4...], 证明是线性启动的


if __name__ == "__main__":
    test_order()
