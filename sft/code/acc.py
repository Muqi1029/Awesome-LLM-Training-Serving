import math
import os

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from prepare_data import get_dataloader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from utils import load_model_and_tokenizer, seed_everything


def save_checkpoint(model, accelerator, config):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        config["output_dir"],
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=False,
    )


@hydra.main(config_path="../config", config_name="train")
def main(config: DictConfig):
    print(config)
    seed_everything(config["seed"])
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )
    accelerator.init_trackers(
        project_name=f"sft_{config['model_name'].split('/')[-1]}",
        config=dict(config),
    )
    if accelerator.is_main_process:
        if config["output_dir"] is not None:
            os.makedirs(config["output_dir"], exist_ok=True)

    if accelerator.is_main_process:
        print(f"Using {config['model_name']} for training")

    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # dataset
    if config["test"]:
        #  Use a smaller subset of data for testing
        test_config = config.copy()
        test_config["max_samples"] = 100  # Limit to 100 samples for testing
        train_dataloader = get_dataloader(test_config, tokenizer)
    else:
        train_dataloader = get_dataloader(config, tokenizer)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=config["learning_rate"]
    )
    total_steps = config["num_epochs"] * math.ceil(
        len(train_dataloader) / config["gradient_accumulation_steps"]
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # accelerate prepare
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    model.train()
    steps = 0
    for epoch in tqdm(range(config["num_epochs"]), desc="Epochs"):
        for batch in tqdm(train_dataloader, desc="Batches"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                steps += 1

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": steps,
                        },
                        step=steps,
                    )

            if steps % config["save_steps"] == 0:
                if accelerator.is_main_process:
                    print(f"Saving checkpoint at step {steps}")
                accelerator.wait_for_everyone()
                save_checkpoint(model, accelerator, config)


if __name__ == "__main__":
    main()
