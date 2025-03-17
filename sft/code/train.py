import logging
import random
from datetime import datetime
from functools import partial

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from sft.code.utils import load_model_and_tokenizer


@hydra.main(config_path="../config", config_name="training_args.yaml")
def train(config: DictConfig):
    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    ds = load_dataset(config["dataset"], split="train")
    if config["test"]:
        random.seed(config["seed"])
        ds = ds.select(random.sample(range(len(ds)), 100))

    map_function = partial(preprocess_dataset, tokenizer=tokenizer, config=config)

    preprocessed_ds = ds.map(
        map_function,
        num_proc=config["num_proc"],
        remove_columns=ds.column_names,
        desc="processing dataset",
    )

    training_arguments = TrainingArguments(
        output_dir=config["output_dir"],
        save_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        fp16=True,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )
    ds_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=ds_collator,
        train_dataset=preprocessed_ds,
    )
    trainer.train()


if __name__ == "__main__":
    train()
