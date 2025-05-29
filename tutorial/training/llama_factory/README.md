# LLaMA Factory

Use a `yaml` file to configure the training process.

## SFT Training

### Dataset Preparation

Since the training logic is well written in the `LLaMA-Factory`, as a user, the core of the work is to prepare a dataset.

How to prepare a dataset?

1. Refer to [LLaMA-Factory/data/README.md](LLaMA-Factory/data/README.md)

The datasets are all organized in `List[Dict]` json format. Your task is to match the field of each sample(Dict) with the required fields in the `dataset_info.json` file.

The following is a simplied explanation that helps you understand how to prepare a SFT dataset quickly and add information into the `dataset_info.json` file.

There are two templates:

- `alpaca`(default):
  - `instruction` str: The instruction to be used for training.
  - `input` str: The input to be used for training.
  - `output` str: The output to be used for training.

- `sharegpt`: The dataset is in the sharegpt format.
  - `messages` List[Dict[str, str]]: The conversation

2. In the `dataset_info.json`, add a new dataset description.
Assuming a sample in the dataset is like this:

**alpaca format**

```json
{
    "instruction": "user instruction in the first round",
    "input": "model response in the first round",
    "output": "user instruction in the second round",
    "system": "system prompt (default: None)"
}
```

You should set the `columns` field as follows:

```json
"{self_prepared_dataset_name}": {

  # self prepared dataset file name
  "file_name": "{self_prepared_dataset_path}"

  # if you want to use a dataset from the internet, you need to set `hf_hub_url` field to the url of the dataset.
  "hf_hub_url": "{hf_hub_url}",

  "formatting": "alpaca", # default

  "columns": { # this is the matching field of each sample
    "prompt": "instruction (default: instruction)",
    "query": "input (default: input)",
    "response": "output (default: output)",

    "system": "system (default: None)", # add a `system prompt`
    "history": [ # history messages in the `alpaca` format, but this will also be learned by the model
        ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
}
```

> Note: In SFT, the `instruction` column will be concatenated with the `input` column and used as the `user prompt`, then the user prompt would be `{prompt}\n{query}`. The `output` column represents the model response. The `response` field is the `assistant prompt`.

**sharegpt format**

Assuming a sample in the dataset is like this:

```json
{
    "messages": [
        {"role": "user", "content": "user instruction in the first round (optional)"},
        {"role": "assistant", "content": "model response in the first round (optional)"}
    ]
}
```

You should set the `columns` field as follows:

```json
    "{self_prepared_dataset_name}": {
        "file_name": "{self_prepared_dataset_path}",
        "hf_hub_url": "{hf_hub_url}", # if you want to use a dataset from the internet, you need to set `hf_hub_url` field to the url of the dataset.
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages (default: conversations)" # List[Dict[str, str]]
        },
        "tags": {

            "role_tag": "role (default: from)", # the key in the message represents the identity.
            "content_tag": "content (default: value)", # the key in the message represents the content.
            "user_tag": "user (default: human)", # the value of the role_tag represents the user.
            "assistant_tag": "assistant (default: gpt)" # the value of the role_tag represents the assistant.
        }
    }
```

### Yaml File

1. Copy a template yaml file: `LLaMA-Factory/examples/train_full/llama3_full_sft.yaml`, rename it as `{model_name}_full_sft.yaml`.

2. Modify the yaml file.

Model

- `model_name_or_path`: The model to be trained.

Method

- `stage`: `sft`

Dataset

- `dataset`: The dataset to be used for training.
- `template`: The template to be used for training.
- `cutoff_len`: The cutoff length to be used for training.
- `max_samples`: The maximum number of samples to be used for training.

Output

- `output_dir`: The output directory saving the checkpoint
- `report_to`: The report to be used for training.

## Reward Training

TODO
