import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function


def simple_demo():
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )


def memory_demo():
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof:
        model(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


def main():
    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]
    elif torch.xpu.is_available():
        device = "xpu"
        activities += [ProfilerActivity.XPU]
    else:
        print(
            "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
        )
        import sys

        sys.exit(0)

    sort_by_keyword = device + "_time_total"

    model = models.resnet18().to(device)
    inputs = torch.randn(5, 3, 224, 224).to(device)

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

    # export traces
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    # main()
    memory_demo()
