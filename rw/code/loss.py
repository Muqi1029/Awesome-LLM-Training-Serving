from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


def reward_model_loss(
    model: nn.Module,
    chosen_batch: Dict[str, Tensor],
    rejected_batch: Dict[str, Tensor],
    temperature: float = 1.0,
    margin: float = 0.0,
) -> Tensor:
    assert set(chosen_batch.keys()) == {
        "input_ids",
        "attention_mask",
    }, "chosen_batch and rejected_batch must contain input_ids and attention_mask"
    chosen_scores = model(**chosen_batch).logits.squeeze(-1)  # [batch_size, ]
    rejected_scores = model(**rejected_batch).logits.squeeze(-1)

    diff = (chosen_scores - rejected_scores - margin) / temperature

    loss = -F.logsigmoid(diff.clamp(min=-50, max=50)).mean(dim=0)

    return loss
