# Model Parallelism

## Why Model Parallelism?

In the context of AI era, the model size is becoming larger and larger, which is not able to be fitted into a single GPU.

Therefore, we need to use Model Parallelism to split the model across multiple GPUs to avoid the memory limit.

Another reason is we want to speed up the inference of the model utilizing our existing hardware as much as possible.

## What is Model Parallelism?

The Model Parallelism is a technique to split the model across multiple GPUs to avoid the memory limit.

There is mainly the following types of model parallelism:

1. Tensor Parallel
2. Pipeline Parallel
3. Sequence Parallel
