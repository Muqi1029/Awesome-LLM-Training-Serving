# PyTorch 分布式进程组

## 为什么需要进程组？

在分布式训练场景中，PyTorch 默认会将所有进程组成一个全局默认组。但在某些场景下，我们需要将默认组拆分为更小的子组，例如将张量并行组（Tensor Parallel Group）和数据并行组（Data Parallel Group）从默认组中分离，以实现更优的并行性能。

## 如何使用进程组？

在所有的分布式集合通信操作中，都可以通过 group 参数指定参与通信的进程组。

1. 注册进程组

使用 new_group 创建自定义进程组：

torch.distributed.new_group(ranks, backend)

2. 在通信操作中使用进程组

将创建的进程组传入分布式操作：

torch.distributed.all_gather(tensor_list, data, group=group)

具体实现示例可参考：[tutorial/distributed/group/group_list.py](group_list.py)

## 参考资料

- [vLLM GroupCoordinator](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py#L174)
