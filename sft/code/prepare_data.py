from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import DataCollatorForSeq2Seq

prompt_template = """
{question}
"""

response_template = """
# Thinking\n\n
{cot}\n\n
## Final Response\n\n
{response}
"""

alpaca_eval_prompt_template = """
{question}
"""

alpaca_eval_response_template = """
<thinking>\n\n
{cot}\n\n
<final_response>\n\n
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


class LMSYS_CHAT_1M_Dataset(Dataset):
    def __init__(self, config, tokenizer):
        self.dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        self.dataset = self.dataset.map(
            lambda x: preprocess_chat_dataset(x, tokenizer, config),
            num_proc=config["num_proc"],
            batched=True,
            remove_columns=self.dataset.column_names,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


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
            num_proc=config["num_proc"],
            remove_columns=self.dataset.column_names,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


alpaca_eval_prompt_template = """
### Instruction:
{instruction}
"""

alpaca_eval_response_template = """
### Response:
{output}
"""


def map_alpaca_eval_dataset(example, tokenizer, config):
    convs = [
        {
            "role": "user",
            "content": alpaca_eval_prompt_template.format(
                instruction=example["instruction"]
            ),
        },
        {
            "role": "assistant",
            "content": alpaca_eval_response_template.format(output=example["output"]),
        },
    ]

    output = tokenizer.apply_chat_template(convs, add_generation_prompt=True)

    labels = output.copy()

    user_message = tokenizer.apply_chat_template([convs[0]], add_generation_prompt=True)
    user_message_len = len(user_message)

    labels[:user_message_len] = [-100] * user_message_len

    if len(output) > config["max_length"]:
        output = output[: config["max_length"]]
        labels = labels[: config["max_length"]]

    return {"input_ids": output, "labels": labels}


class AlpacaEvalDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.dataset = load_dataset(config["data_path"], split="test")

        # map input output template
        self.dataset = self.dataset.map(
            partial(map_alpaca_eval_dataset, tokenizer=tokenizer),
            num_proc=config["num_proc"],
            batched=False,
            remove_columns=self.dataset.column_names,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_dataloader(config, tokenizer):
    if config["dataset"] == "lmsys/lmsys-chat-1m":
        ds = LMSYS_CHAT_1M_Dataset(config, tokenizer)
    elif config["dataset"] == "tatsu-lab/alpaca_eval":
        ds = AlpacaEvalDataset(config, tokenizer)
    else:
        ds = SFTDataset(config, tokenizer)

    if config["test"]:
        ds = Subset(ds, range(config["max_samples"]))
    dataloader = DataLoader(
        dataset=ds,
        batch_size=config["per_device_train_batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_proc"],
        drop_last=config["drop_last"],
        pin_memory=True,
    )

    return dataloader
