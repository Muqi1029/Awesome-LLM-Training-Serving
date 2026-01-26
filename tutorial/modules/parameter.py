from collections import OrderedDict

import torch
from torch import nn


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, 3, 1), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(OrderedDict(sublayer2=nn.Linear(2, 3)))

        self.unlearnable_param = torch.randn(20, 3)
        self.learnable_param = nn.Parameter(torch.randn(20, 3))
        self.register_parameter(
            "register_param", nn.Parameter(torch.randn(20, 3))
        )  # = self.register_param = nn.Parameter(torch.randn(20, 3))
        self.register_buffer("buffer", self.unlearnable_param)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def main():
    model = TestModule()
    p = model.parameters()
    print(f"{type(p)=}")
    for p_item in p:
        print(p_item.shape)

    # divider
    print("named_parameters".center(80, "-"))
    n_p = model.named_parameters()
    print(f"{type(n_p)=}")
    for name, n_p_item in n_p:
        print(f"{name:35} {n_p_item.shape}")

    print("modules | named_modules".center(80, "-"))
    print("modules:", list(model.modules()))
    print("named_modules:", list(model.named_modules()))

    print(f"children | named_children".center(80, "-"))
    print("children:", list(model.children()))
    print("named_children:", list(model.named_children()))

    print(f"model state_dict".center(80, "-"))
    state_dict = model.state_dict()
    print(f"{type(state_dict)=}")
    for name, param_or_buffer in state_dict.items():
        print(
            f"name={name:30} type={type(param_or_buffer)} shape={param_or_buffer.shape}"
        )


if __name__ == "__main__":
    main()
