from argparse import ArgumentParser

from openai import OpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "get current weather",
            "parameters": {
                "type": "object",
                "required": ["city", "state", "unit"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
            },
        },
    }
]


def stream_chat(args):
    extra_body = {
        "chat_template_kwargs": (
            {"enable_thinking": False} if args.disable_thinking else {}
        ),
        "separate_reasoning": True,
    }

    responses = client.chat.completions.create(
        model="Qwen3",
        messages=[{"role": "user", "content": args.prompt}],
        extra_body=extra_body,
        stream=True,
        temperature=0.6,
        top_p=0.95,
        # min_p=0,
    )
    # for choice in responses.choices:
    #     print(choice.message.content, end='')
    for chunk in responses:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
    print()


def tool_request(args):
    extra_body = {
        "chat_template_kwargs": (
            {"enable_thinking": False} if args.disable_thinking else {}
        ),
    }

    responses = client.chat.completions.create(
        model="Qwen3",
        messages=[{"role": "user", "content": args.prompt}],
        tools=tools,
        extra_body=extra_body,
    )
    for choice in responses.choices:
        print(choice.message.content)
        if choice.message.tool_calls:
            print(choice.message.tool_calls)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--tool", action="store_true")
    args = parser.parse_args()
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="Bear None")
    if args.stream:
        stream_chat(args)
    elif args.tools:
        tool_request(args)
