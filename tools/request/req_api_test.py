# Test Generate API
import os
from argparse import ArgumentParser

import requests
from openai import OpenAI
from transformers import AutoTokenizer

model_path = os.environ["DEEPSEEK_R1_DISTILL_QWEN_7B"]
base_url = os.environ["BASE_URL"]
api_key = os.environ["API_KEY"]

client = OpenAI(
    base_url=base_url,
    api_key=os.environ.get("API_KEY", "Bearer JustKeepMe"),
)


def non_streaming_response(client: OpenAI):
    response = client.responses.create(model=model_path, input="What is 1+3?")
    print(response.model_dump_json(indent=2))


def non_stream_chat(text: str):
    response = requests.post(
        url=f"{base_url}/generate",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "text": text,
            "require_reasoning": True,
            "sampling_params": {"max_new_tokens": 1024},
        },
    )
    print(response.json())


def stream_chat(text):
    response_stream = requests.post(
        url=f"{base_url}/generate",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "text": text,
            "require_reasoning": True,
            "sampling_params": {"max_new_tokens": 1024},
            "stream": True,
        },
        stream=True,
    )
    response_stream.raise_for_status()
    for line in response_stream.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        print(line_str)


def main(args):
    if args.api == "response":
        non_streaming_response(client)
    elif args.api == "generate":
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        messages = [{"role": "user", "content": "What is 1+3?"}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        stream_chat(text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--api", choices=["response", "generate"], help="api form for test"
    )
    parser.add_argument("--stream", action="store_true", help="")
    args = parser.parse_args()
    main(args)
