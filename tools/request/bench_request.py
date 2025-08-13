"""
this request is specailly written for Qwen3-235B-A22B-FP8 Model
"""

import os
from argparse import ArgumentParser

import requests
from transformers import AutoTokenizer

# global args
input_len = 127 * 1024
output_len = 1 * 1024 - 1
port = 8888
host = "0.0.0.0"
base_url = f"http://{host}:{port}/v1"
api_key = "just keep me"
model_path = os.environ.get("QWEN3235BA22BFP8", "")


def prepare_prompt(tokenizer, input_len):
    prompt = "hello, world"
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_token_ids)
    if prompt_len > input_len:
        input_token_ids = prompt_token_ids[:input_len]
    else:
        ratio = (input_len - 1) // prompt_len + 1
        input_token_ids = (prompt_token_ids * ratio)[:input_len]
    print(f"There are totally {len(input_token_ids)} length input_ids")
    adjusted_prompt = tokenizer.decode(input_token_ids)
    return adjusted_prompt


def openai_request(prompt, model_path: str):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    outputs = client.completions.create(
        prompt=prompt,
        model=model_path,
        temperature=0,
        max_tokens=output_len,
        n=1,
        stop=None,
    )
    return outputs


def raw_request(prompt, model_path):
    outputs = requests.post(
        url=base_url,
        headers={"Authorization": f"Bear: {api_key}"},
        json={
            "prompt": prompt,
            "model": model_path,
            "temperature": 0,
            "max_tokens": output_len,
            "best_of": 1,
            "ignore_eos": True,
        },
    )
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="openai",
        choices=["openai", "raw"],
        help="the mode to send request",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = prepare_prompt(tokenizer, input_len=input_len)
    if args.mode == "openai":
        outputs = openai_request(prompt, model_path)
    else:
        outputs = raw_request(prompt, model_path)

    for output in outputs.choices:
        print(output.text)


if __name__ == "__main__":
    main()
