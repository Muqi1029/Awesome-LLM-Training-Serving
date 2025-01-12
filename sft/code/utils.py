from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def preprocess_dataset(messages, tokenizer, config, INGORE_INDEX=-100):
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
