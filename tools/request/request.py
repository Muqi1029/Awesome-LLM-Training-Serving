import argparse
import json
from pprint import pprint

import openai
import requests

tool_select_name = {
    "type": "function",
    "function": {
        "name": "select_name",
        "description": "select a name",
        "additionalproperties": "false",
        "strict": "true",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "the name",
                    "enum": ["Muqi Li", "Muqi1029"],
                },
                "age": {
                    "type": "integer",
                    "description": "a number from 0 to 23, which represent the person's age",
                    "minimum": 0,
                    "maximum": 23,
                },
            },
        },
        "required": ["name", "age"],
    },
}

tools = [tool_select_name]

ebnf_content = """
root ::= city | description

city ::= "London" | "Pairis" | "Berlin" | "Rome"
description ::= city " is " status

status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

# ebnf = """
# root ::= number " + " number " = " number
#
# number ::= integer | float
#
# integer ::= digit+
# float ::= number "." number
# digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
# """


def info_print(payload, url):
    print("=" * 80)
    print(f"Sending to {url=}")
    pprint(f"{payload=}")


def http_request(args):
    url = f"{args.base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"}

    # construct payload
    payload = {}
    if args.disable_separate_reasoning:
        payload["separate_reasoning"] = False

    if not args.msg_path and not args.input_ids_path and not args.payload_path:
        # if there is no msg path, input_ids path, payload path
        payload["messages"] = [{"role": "user", "content": args.user_content}]
        if args.ebnf:
            payload["ebnf"] = ebnf_content
        elif args.tools:
            payload["tools"] = tools
    else:
        if args.msg_path:
            # read message path
            with open(args.msg_path, mode="r", encoding="utf-8") as f:
                payload["messages"] = json.load(f)
        elif args.payload_path:
            # read payload path
            with open(args.payload_path, mode="r", encoding="utf-8") as f:
                payload = json.load(f)
        elif args.input_ids_path:
            # read input_ids path
            with open(args.input_ids_path, mode="r", encoding="utf-8") as f:
                input_ids = json.load(f)
            from transformers import AutoTokenizer

            assert (
                args.tokenizer_path
            ), f"tokenizer_path must be provided in the input-ids-path"
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            prompt = tokenizer.decode(input_ids)
            print(f"{prompt=}")
            payload["prompt"] = prompt
            url = f"{args.base_url}/v1/completions"

    if args.disable_stream:
        info_print(payload, url)
        res = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        try:
            res.raise_for_status()
            print(res.json())
        except Exception:
            print(
                f"\033[41m Request Error, Status Code={res.status_code}, Reason: {res.text} \033[0m"
            )
    else:
        payload["stream"] = True
        info_print(payload, url)
        res = requests.post(
            url=url,
            headers=headers,
            json=payload,
            stream=True,
        )
        for line in res.iter_lines():
            if not line:
                continue
            decoded_line = line.decode("utf-8")

            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                if data_str.strip() == "[DONE]":
                    print("\n[DONE]")
                    break

                try:
                    chunk = json.loads(data_str)

                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if reasoning_content := delta.get("reasoning_content", ""):
                            print(reasoning_content, end="", flush=True)

                        if content := delta.get("content", ""):
                            print(content, end="", flush=True)

                        if tool_calls := delta.get("tool_calls", ""):
                            tc = tool_calls[0]
                            if func_name := tc.get("function", {}).get("name"):
                                print(f"\n\n[Tool Call Detected]: Function={func_name}")
                                print("Arguments: ", end="", flush=True)
                            if func_arg := tc.get("function", {}).get("arguments"):
                                print(func_arg, end="", flush=True)

                except json.JSONDecodeError:
                    continue


def openai_request(args):
    client = openai.OpenAI(
        base_url=f"{args.base_url}/v1/chat/completions",
        api_key=args.api_key,
    )
    model_id = list(client.models.list())[0].id

    extra_body = {}
    if args.enbf:
        extra_body["ebnf"] = ebnf_content

    if args.disable_stream:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "who are you"}],
        )
        print(response)
    else:
        response_stream = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "who are you",
                }
            ],
            model=model_id,
            extra_body=extra_body,
            # extra_body={
            #     "chat_template_kwargs": {"thinking": True}
            # },  # True or False to control model thinking
            # tools=tools,
            # tool_choice = ,
            stream=True,
            # stream_options={"include_usage": True}
        )
        for chunk in response_stream:
            choices = chunk.choices
            if choices:
                choice = choices[0]
                if reasoning_content := choice.delta.reasoning_content:
                    print(reasoning_content, end="", flush=True)
                if content := choice.delta.content:
                    print(content, end="", flush=True)
                if tool_calls := choice.delta.tool_calls:
                    print(tool_calls[0], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--base-url", type=str, default="http://localhost:8888")
    parser.add_argument("--api-key", type=str, default="JustKeepMe")
    parser.add_argument("--disable-stream", action="store_true")
    parser.add_argument(
        "--backend", type=str, default="http", choices=["http", "openai"]
    )
    parser.add_argument("--user-content", type=str, default="Who are you")
    parser.add_argument("--tokenizer-path", type=str, help="The path of tokenizer path")

    parser.add_argument(
        "--disable-separate-reasoning",
        action="store_true",
        help="Whether to separate reasoning",
    )

    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--ebnf", action="store_true", help="Constrained Decoding for EBNF format"
    )
    mutex_group.add_argument("--tools", action="store_true", help="Add tool")
    mutex_group.add_argument("--msg-path", type=str, help="The path of messages")
    mutex_group.add_argument("--payload-path", type=str, help="The path of payload")
    mutex_group.add_argument("--input-ids-path", type=str, help="The path of input_ids")

    args = parser.parse_args()
    if args.backend == "http":
        http_request(args)
    else:
        openai_request(args)
