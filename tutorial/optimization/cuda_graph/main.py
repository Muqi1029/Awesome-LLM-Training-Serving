import torch
from torch import nn
from time import perf_counter
import sys


class Config:
    forward_times = 100

    batch_size = 20
    hidden_size = 10
    up_ratio = 4


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_size * config.up_ratio)
        self.rms_norm = nn.RMSNorm(config.hidden_size * config.up_ratio)
        self.down_proj = nn.Linear(config.hidden_size * config.up_ratio, config.hidden_size)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.rms_norm(x)
        x = self.down_proj(x)
        return x

def use_cuda_graph(model, config, inputs_for_record, inputs_for_run):
    g = torch.cuda.CUDAGraph()

    static_outputs = torch.empty(config.batch_size, config.hidden_size).to("cuda")

    # warm up
    _ = model(inputs_for_record)

    # start recording
    with torch.cuda.graph(g):
        static_outputs.copy_(model(inputs_for_record))
    
    # run the graph
    inputs_for_record.copy_(inputs_for_run) # prepare the new inputs
    start = perf_counter()
    for _ in range(config.forward_times):
        g.replay()
    time = perf_counter() - start

    print(f"cuda graph time:\t{time}")
    return static_outputs

def use_normal(model, config, inputs_for_record, inputs_for_run):
    inputs_for_record.copy_(inputs_for_run) # prepare the new inputs

    time = perf_counter()
    for _ in range(config.forward_times):
        outputs = model(inputs_for_record)
    time = perf_counter() - time
    print(f"normal time:\t{time}")
    return outputs

if __name__ == "__main__":
    # script: python main.py [cuda_graph, normal, all]
    assert torch.cuda.is_available(), "cuda is not available"
    assert len(sys.argv) == 2 and sys.argv[1] in ["cuda_graph", "normal", "all"], "usage: python main.py [cuda_graph, normal, all]"

    config = Config()
    torch.cuda.manual_seed(42)

    inputs_for_record = torch.randn(config.batch_size, config.hidden_size).to("cuda")
    inputs_for_run = torch.randn(config.batch_size, config.hidden_size).to("cuda")

    model = Model(config).to("cuda")
    if sys.argv[1] == "cuda_graph":
        use_cuda_graph(model, config, inputs_for_record, inputs_for_run)
    elif sys.argv[1] == "normal":
        use_normal(model, config, inputs_for_record, inputs_for_run)
    elif sys.argv[1] == "all":
        output_cuda_graph = use_cuda_graph(model, config, inputs_for_record, inputs_for_run)
        output_normal = use_normal(model, config, inputs_for_record, inputs_for_run)
        assert torch.allclose(output_cuda_graph, output_normal), "output of cuda graph and normal are not the same"
