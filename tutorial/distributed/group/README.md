[中文版本](./README_zh.md)

# Torch Distributed Group

## Why need group?

In distributed setting, by default PyTorch uses all the processes to form a default group.

However, in some cases, we may want to split the default group into smaller groups. For example, we want to split the tensor parallel group and data parallel group from the default group, in order to acheive better performance.

## How to use group?

In all distibuted collective operations, we can pass the group to the operation to specify the group of processes.

1. Register a group

```python
torch.distributed.new_group(ranks, backend)
```

2. Use the group in the operation

```python
torch.distributed.all_gather(tensor_list, data, group=group)
```

There exists a concrete demo in [tutorial/distributed/group/group_list.py](group_list.py).

## References

- [vLLM GroupCoordinator](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py#L174)
