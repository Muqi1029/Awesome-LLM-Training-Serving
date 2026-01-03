"""ref: https://github.com/gpu-mode/profiling-cuda-in-torch/blob/main/pt_profiler.py"""

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

# ## Default way to use profiler
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     for _ in range(10):
#         a = torch.square(torch.randn(10000, 10000).cuda())

# prof.export_chrome_trace("trace.json")


## With warmup and skip
# https://pytorch.org/docs/stable/profiler.html


# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")


with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    schedule=schedule(wait=1, warmup=1, active=2, repeat=2),
    on_trace_ready=trace_handler,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
) as p:
    for iter in range(10):
        torch.square(torch.randn(10000, 10000).cuda())
        # send a signal to the profiler that the next iteration has started
        p.step()
