import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn


@pytest.fixture(scope="module")
def input_():
    bsz = 10
    seq_len = 2
    hidden_size = 128
    return torch.rand(bsz, seq_len, hidden_size)


def test_mha(input_):
    pass


def test_mla(input_):
    pass


def test_gqa_shape(input_: torch.Tensor):
    bsz, seq_len, hidden_size = input_.size()

    n_groups = 2
    num_heads = 4
    head_dim = 32
    assert head_dim * num_heads == hidden_size
    assert num_heads % n_groups == 0

    # prepare weights
    w_q = nn.Linear(hidden_size, num_heads * head_dim)
    w_k = nn.Linear(hidden_size, n_groups * head_dim)
    w_v = nn.Linear(hidden_size, n_groups * head_dim)

    w_o = nn.Linear(hidden_size, hidden_size)

    q = w_q(input_).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = w_k(input_).view(bsz, seq_len, n_groups, head_dim)
    v = w_v(input_).view(bsz, seq_len, n_groups, head_dim)

    num_repeats = num_heads // n_groups
    repeat_k = torch.repeat_interleave(k, repeats=num_repeats, dim=2).transpose(1, 2)
    repeat_v = torch.repeat_interleave(v, repeats=num_repeats, dim=2).transpose(1, 2)

    scores = q @ repeat_k.transpose(-1, -2) / math.sqrt(head_dim)

    scores = F.softmax(scores, dim=-1)

    o = (scores @ repeat_v).transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    assert w_o(o).size() == (bsz, seq_len, hidden_size)
