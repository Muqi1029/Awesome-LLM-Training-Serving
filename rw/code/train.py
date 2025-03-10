import yaml
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Qwen2ForSequenceClassification,
)

from rw.code.loss import reward_model_loss
from rw.code.prepare_data import get_dataloader


def main():
    with open(file="config/hh.yaml", mode="r") as fp:
        config = yaml.safe_load(fp)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"], num_labels=1, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    dataloader = get_dataloader(config, tokenizer)

    no_weight_decay = ["bias", "LayerNorm.weight"]
    group_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_weight_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_weight_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params=group_params, lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        for batch_idx, batch in enumerate(dataloader):
            for k, v in batch["chosen_inputs"].items():
                batch["chosen_inputs"][k] = v.to(model.device)
            for k, v in batch["rejected_inputs"].items():
                batch["rejected_inputs"][k] = v.to(model.device)
            l = reward_model_loss(
                model=model,
                chosen_batch=batch["chosen_inputs"],
                rejected_batch=batch["rejected_inputs"],
            )
            l.backward()
            if (batch_idx + 1) % config[
                "gradient_accumulation_steps"
            ] == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                print(l.item())
    model.save_pretrained(
        f"models/rw_{config['model_name_or_path'].split('/')[-1]}_{config['epochs']}e"
    )


if __name__ == "__main__":
    main()
