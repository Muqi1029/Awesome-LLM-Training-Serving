import logging
from datetime import datetime
from functools import partial

import yaml
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from sft.code.utils import load_model_and_tokenizer, preprocess_dataset

config = yaml.Loader(open("config/training_args.yaml"))
print(config)
logging.basicConfig(
    filename=f"logs/{config['model_name']}_{datetime()}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s",
)


def train():
    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")

    map_function = partial(preprocess_dataset, tokenizer=tokenizer)

    preprocessed_ds = ds.map(
        map_function,
        batched=True,
        num_proc=8,
        remove_columns=ds.column_names,
        desc="processing dataset",
    )

    training_arguments = TrainingArguments(
        output_dir=config["output_dir"],
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
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
