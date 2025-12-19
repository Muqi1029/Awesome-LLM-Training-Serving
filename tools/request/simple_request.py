import argparse
import json

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


def http_request(args):
    if args.disable_stream:
        res = requests.post(
            url=f"{args.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {args.api_key}"},
            json={
                "messages": [{"role": "user", "content": "Who are you?"}],
            },
        )
        res.raise_for_status()
        print(res.json())
    else:
        res = requests.post(
            url=f"{args.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {args.api_key}"},
            json={
                "messages": [{"role": "user", "content": "Who are you?"}],
            },
            stream=True,
        )
        for line in res.iter_lines():
            if line:
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
    args = parser.parse_args()
    if args.backend == "http":
        http_request()
    else:
        openai_request()
