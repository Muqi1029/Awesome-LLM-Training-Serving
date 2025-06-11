import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GQAAttnConfig:
    hidden_dim: int  # Dimension of the input and output features
    head_dim: int  # Dimension of each attention head
    num_heads: int  # Number of query heads
    num_groups: int  # Number of key/value head groups (num_groups <= num_heads)

    def __post_init__(self):
        if self.num_heads % self.num_groups != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_groups ({self.num_groups})"
            )
        if self.num_groups > self.num_heads:
            raise ValueError(
                f"num_groups ({self.num_groups}) cannot be greater than num_heads ({self.num_heads})"
            )


class GQAAttention(nn.Module):
    def __init__(self, config: GQAAttnConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_groups = config.num_groups

        self.num_heads_per_group = self.num_heads // self.num_groups

        self.q_proj_dim = self.num_heads * self.head_dim
        self.kv_proj_dim = self.num_groups * self.head_dim

        # W_QKV projects the input to Q, K, and V
        # Q will have q_proj_dim, K and V will have kv_proj_dim
        self.W_QKV = nn.Linear(
            self.hidden_dim, self.q_proj_dim + 2 * self.kv_proj_dim, bias=False
        )  # Common to not use bias in QKV

        # W_O projects the concatenated head outputs back to hidden_dim
        # The input to W_O is q_proj_dim (num_heads * head_dim)
        self.W_O = nn.Linear(
            self.q_proj_dim, self.hidden_dim, bias=False
        )  # Common to not use bias in W_O

    def forward(self, input_: torch.Tensor, attention_mask: torch.Tensor = None):
        bsz, seq_len, _ = input_.size()

        qkv = self.W_QKV(input_)

        q, k, v = torch.split(
            qkv, [self.q_proj_dim, self.kv_proj_dim, self.kv_proj_dim], dim=-1
        )

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_groups, self.head_dim).transpose(1, 2)

        # Handle broadcasting for K and V to match the number of query heads
        # k: (bsz, num_groups, seq_len, head_dim)
        # v: (bsz, num_groups, seq_len, head_dim)
        # We repeat K and V along the head dimension `num_heads_per_group` times
        if self.num_groups != self.num_heads:
            # unsqueeze(2) adds a dimension for num_heads_per_group, then repeat
            # -> (bsz, num_groups, num_heads_per_group, seq_len, head_dim)
            k = k.unsqueeze(2).repeat(1, 1, self.num_heads_per_group, 1, 1)
            v = v.unsqueeze(2).repeat(1, 1, self.num_heads_per_group, 1, 1)
            # Flatten num_groups and num_heads_per_group into a single head dimension
            # (bsz, num_heads, seq_len, head_dim)
            k = k.view(bsz, self.num_heads, seq_len, self.head_dim)
            v = v.view(bsz, self.num_heads, seq_len, self.head_dim)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        # Apply attention mask (if provided)
        if attention_mask is not None:
            # mask has shape (bsz, 1, 1, seq_len) or (bsz, 1, seq_len, seq_len)
            # scores has shape (bsz, num_heads, seq_len, seq_len)
            # Broadcasting will handle it correctly.
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=-1)

        o = attention_probs @ v

        o = o.transpose(1, 2).contiguous()

        o = o.view(bsz, seq_len, self.q_proj_dim)

        output = self.W_O(o)
        return output


# Example Usage:
if __name__ == "__main__":
    print("--- Testing GQAAttention ---")

    # GQA Configuration (e.g., 12 query heads, 3 K/V groups)
    gqa_config = GQAAttnConfig(
        hidden_dim=768,
        head_dim=64,
        num_heads=12,
        num_groups=3,  # Each K/V group serves 12 / 3 = 4 query heads
    )
    gqa_layer = GQAAttention(gqa_config)
    print(f"GQA Layer: {gqa_layer}")
    print(f"  Q projection dim: {gqa_layer.q_proj_dim}")
    print(f"  KV projection dim: {gqa_layer.kv_proj_dim}")
    print(f"  Num heads per group: {gqa_layer.num_heads_per_group}")

    batch_size = 2
    sequence_length = 10
    dummy_input = torch.randn(batch_size, sequence_length, gqa_config.hidden_dim)

    # Example 1: Causal Mask (Lower Triangular)
    # This mask is broadcastable to (bsz, num_heads, seq_len, seq_len)
    causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    causal_mask = (
        torch.tril(causal_mask).unsqueeze(0).unsqueeze(0)
    )  # -> (1, 1, seq_len, seq_len)

    print("\nInput shape:", dummy_input.shape)
    print("Causal Mask shape:", causal_mask.shape)

    output_gqa_causal = gqa_layer(dummy_input, attention_mask=causal_mask)
    print("Output shape (GQA with Causal Mask):", output_gqa_causal.shape)

    # Example 2: Padding Mask
    padding_mask = torch.ones(batch_size, 1, 1, sequence_length, dtype=torch.bool)
    padding_mask[0, :, :, 8:] = 0  # Mask last 2 tokens for batch item 0
    padding_mask[1, :, :, 6:] = 0  # Mask last 4 tokens for batch item 1

    print("\nPadding Mask shape:", padding_mask.shape)
    output_gqa_padding = gqa_layer(dummy_input, attention_mask=padding_mask)
    print("Output shape (GQA with Padding Mask):", output_gqa_padding.shape)

    # Example 3: No Mask
    output_gqa_no_mask = gqa_layer(dummy_input)
    print("\nOutput shape (GQA No Mask):", output_gqa_no_mask.shape)

    # Test MQA (num_groups = 1)
    print("\n--- Testing MQA (GQA with num_groups=1) ---")
    mqa_config = GQAAttnConfig(
        hidden_dim=768,
        head_dim=64,
        num_heads=12,
        num_groups=1,  # All 12 query heads share 1 K/V head
    )
    mqa_layer = GQAAttention(mqa_config)
    print(f"MQA Layer: {mqa_layer}")
    print(f"  Q projection dim: {mqa_layer.q_proj_dim}")
    print(f"  KV projection dim: {mqa_layer.kv_proj_dim}")
    print(f"  Num heads per group: {mqa_layer.num_heads_per_group}")
    output_mqa = mqa_layer(dummy_input)
    print("Output shape (MQA):", output_mqa.shape)

    # Test MHA (num_groups = num_heads)
    print("\n--- Testing MHA (GQA with num_groups=num_heads) ---")
    mha_config_from_gqa = GQAAttnConfig(
        hidden_dim=768,
        head_dim=64,
        num_heads=12,
        num_groups=12,  # 12 query heads, 12 K/V heads (standard MHA)
    )
    mha_layer_from_gqa = GQAAttention(mha_config_from_gqa)
    print(f"MHA (via GQA) Layer: {mha_layer_from_gqa}")
    print(f"  Q projection dim: {mha_layer_from_gqa.q_proj_dim}")
    print(f"  KV projection dim: {mha_layer_from_gqa.kv_proj_dim}")
    print(f"  Num heads per group: {mha_layer_from_gqa.num_heads_per_group}")
    output_mha_via_gqa = mha_layer_from_gqa(dummy_input)
    print("Output shape (MHA via GQA):", output_mha_via_gqa.shape)

    # Test for ValueError (num_heads not divisible by num_groups)
    try:
        invalid_config = GQAAttnConfig(
            hidden_dim=768, head_dim=64, num_heads=11, num_groups=3
        )
    except ValueError as e:
        print(f"\nCaught expected error for invalid config: {e}")

    # Test for ValueError (num_groups > num_heads)
    try:
        invalid_config = GQAAttnConfig(
            hidden_dim=768, head_dim=64, num_heads=12, num_groups=13
        )
    except ValueError as e:
        print(f"\nCaught expected error for invalid config: {e}")
