from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RewardModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self._frozen(encoder)
        self.reward_head = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask=None, pos_ids=None):
        outputs = self.encoder(input_ids, token_type_ids, attention_mask, pos_ids)
        return self.reward_head(outputs.last_hidden_state[:, 0, :])

    def _frozen(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False


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
    }, "chosen_batch和rejected_batch的输入需要包含input_ids和attention_mask"
    chosen_scores = model(**chosen_batch).logits.squeeze(-1)  # [batch_size, ]
    rejected_scores = model(**rejected_batch).logits.squeeze(-1)

    diff = (chosen_scores - rejected_scores - margin) / temperature

    loss = -F.logsigmoid(diff.clamp(min=-50, max=50)).mean(dim=0)

    return loss
