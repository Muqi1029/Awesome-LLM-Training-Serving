import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


model = SimpleModel()
dummy_input = torch.randn(64, 10)

# 确保保存为 .json 文件，而不是 TensorBoard log
# 指定输出文件名，而不是目录
output_trace_file = "./my_trace_us.json"

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.export_chrome_trace(
        output_trace_file
    ),  # 使用 trace_handler 直接保存 JSON
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for step in range(5):
        _ = model(dummy_input)
        prof.step()

print(f"Profiling complete. Data saved to {output_trace_file}")
