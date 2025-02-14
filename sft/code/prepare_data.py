from os import remove
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from functools import partial


prompt_template = """
{question}
"""

response_template = """
# Thinking\n\n
{cot}\n\n
## Final Response\n\n
{response}
"""


def preprocess_chat_dataset(messages, tokenizer, config, INGORE_INDEX=-100):
    messages = messages["conversation"]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    data = {}
    data["input_ids"] = inputs
    data["labels"] = inputs.copy()
    data["attention_mask"] = [1] * len(inputs)

    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = len(tokenizer.apply_chat_template(messages[:idx]))

            if idx < len(messages) - 1 and messages[idx + 1]["role"] == "assistant":
                # if next role is "assistant", include the generation prompt to be ignored
                message_end_idx = len(
                    tokenizer.apply_chat_template(
                        messages[: idx + 1], add_generation_prompt=True
                    )
                )
            else:
                message_end_idx = len(
                    tokenizer.apply_chat_template(messages[: idx + 1])
                )
            # set this message to be ignored
            data["labels"][message_start_idx:message_end_idx] = [INGORE_INDEX] * (
                message_end_idx - message_start_idx
            )
    if "max_length" in config:
        data["input_ids"] = data["input_ids"][: config["max_length"]]
        data["labels"] = data["labels"][: config["max_length"]]
        data["attention_mask"] = data["attention_mask"][: config["max_length"]]
    return data


def apply_input_output_template(example):
    convs = []
    prompt = prompt_template.format(question=example["Question"])
    convs.append({"role": "user", "content": prompt})
    res = response_template.format(
        cot=example["Complex_CoT"], response=example["Response"]
    )
    convs.append({"role": "assistant", "content": res})
    return {"conversation": convs}


class SFTDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.dataset = load_dataset(config["data_path"], split="train")

        # map input output template
        self.dataset = self.dataset.map(
            apply_input_output_template, num_proc=config["num_proc"]
        )

        # map tokenize
        map_func = partial(
            preprocess_chat_dataset,
            tokenizer=tokenizer,
            config=config,
        )
        self.dataset = self.dataset.map(
            map_func,
            num_proc=None if "num_proc" in config else config["num_proc"],
            remove_columns=self.dataset.column_names,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_dataloader(config, tokenizer):
    ds = SFTDataset(config, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    dataloader = DataLoader(
        dataset=ds,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=data_collator,
    )
    return dataloader
