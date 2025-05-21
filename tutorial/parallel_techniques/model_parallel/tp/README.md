# Tensor Parallelism

## What is Tensor Parallelism?

Tensor Parallelism is a technique to parallelize the computation of a tensor operation across multiple GPUs, which is also `gemm` operation.

## Application of Tensor Parallelism in LLMs

This part mainly introduces how TP is applied in the forward of transformer layers.

For the ease of understanding, let me introduce the basic usage of Matrix Multiplication using TP.

Which is mainly classified into two cases:

1. Row Parallelism

> split the rows of the matrix.

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = False):
        super().__init__()
        self.world_size = dist.get_world_size()  # get the world size
        assert (
            in_features % self.world_size == 0
        ), f"in_features must be divisible by world_size, but got {in_features=}, {self.world_size=}"
        part_in_features = in_features // self.world_size
        self.weight = nn.Parameter(torch.empty(out_features, part_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.world_size > 1:
            dist.all_reduce(y)
        return y
```

2. Column Parallelism

> split the columns of the matrix.

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = False):
        super().__init__()
        world_size = dist.get_world_size()
        assert (
            out_features % world_size == 0
        ), f"out_features should be divisible by world_size, but got {out_features=}, {world_size=}"
        part_out_features = out_features // world_size
        self.weight = nn.Parameter(torch.empty(part_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(part_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        return y
```

### Embedding Layer

### MLP Layer

### Attention Layer

## Benchmark

In this part, we will benchmark the performance of the TP in the different backends(gloo, nccl) with different hardware to indicate the performance of the TP.
> The benchmark file is [here](./benchmark.py).

### GLOO (CPU)

### NCCL (GPU)

#### 3090

#### A40

## References

- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
