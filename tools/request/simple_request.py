import argparse

import openai
import requests


def http_request(args):
    res = requests.post(
        url=f"{args.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json={
            "messages": [{"role": "user", "content": "Who are you?"}],
        },
    )
    res.raise_for_status()
    print(res.json())


def openai_request(args):
    client = openai.OpenAI(
        base_url=f"{args.base_url}/v1/chat/completions",
        api_key=args.api_key,
    )

    model_id = list(client.models.list())[0].id
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "who are you"}],
    )
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--base-url", type=str, default="http://localhost:8888")
    parser.add_argument("--api-key", type=str, default="JustKeepMe")
    parser.add_argument(
        "--backend", action="store_true", default="http", choices=["http", "openai"]
    )
    args = parser.parse_args()
    if args.backend == "http":
        http_request()
    else:
        openai_request()
