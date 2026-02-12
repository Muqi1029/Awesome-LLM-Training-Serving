import torch
from torch import nn


def rms_norm(x, weight, eps=1e-6):
    rms_inv = torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + eps)
    return x * rms_inv * weight


def rms_norm_naive(x, weight, eps=1e-6):
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def test_rms_norm_forward():
    torch.manual_seed(0)

    batch, seq, hidden = 2, 4, 8
    x = torch.randn(batch, seq, hidden)
    weight = torch.randn(hidden)

    y = rms_norm(x, weight)
    y_ref = rms_norm_naive(x, weight)

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    print("✅ Forward test passed")


if __name__ == "__main__":
    test_rms_norm_forward()
