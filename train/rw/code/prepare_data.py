from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


def preprocess_data(example, tokenizer, config):
    data = {}

    tokenized_chosen_inputs = tokenizer(
        example["chosen"],
        truncation=True,
        padding=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )
    data["chosen_input_ids"] = tokenized_chosen_inputs["input_ids"]

    tokenized_rejected_inputs = tokenizer(
        example["rejected"],
        max_length=config["max_length"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    data["rejected_input_ids"] = tokenized_rejected_inputs["input_ids"]

    return data


class RewardDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.data = load_dataset(config["data_path"], split="train")
        self.tokenizer = tokenizer
        self.preprocess_data = partial(
            preprocess_data, tokenizer=tokenizer, config=config
        )
        self.data = self.data.map(
            self.preprocess_data,
            batched=True,
            num_proc=config["num_proc"],
            remove_columns=self.data.column_names,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch: list, tokenizer: PreTrainedTokenizerFast, config: dict) -> dict:
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]

    padded_batch = {
        "chosen_inputs": tokenizer.pad(
            {"input_ids": chosen_input_ids}, padding="longest", return_tensors="pt"
        ),
        "rejected_inputs": tokenizer.pad(
            {"input_ids": rejected_input_ids}, padding="longest", return_tensors="pt"
        ),
    }

    return padded_batch


def get_dataloader(config, tokenizer):
    dataset = RewardDataset(config, tokenizer)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=partial(collate_fn, tokenizer=tokenizer, config=config),
    )
