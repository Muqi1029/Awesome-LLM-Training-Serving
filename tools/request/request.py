"""
Usage:
1. Send a "who are you" Request: python request.py --base-url <default: http://127.0.0.1:8888> --api-key <default: JustKeepMe> --backend <default: http>
2. Send a payload Requests: python requests.py --payload-path <path of payload>
"""

import argparse
import json
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict

import openai
import requests

json_schema_response_format = {
    "name": "require_named",
    "description": "a schema for the response format",
    "schema": {
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
        "required": ["name", "age"],
        "additionalProperties": False,
    },
    "strict": True,
}

tool_select_name = {
    "type": "function",
    "function": {
        "name": "select_name",
        "description": "select a name",
        "additionalproperties": False,
        "strict": True,
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


class Color(Enum):
    LIGHT_CYAN = auto()
    LIGHT_GREEN = auto()
    LIGHT_YELLOW = auto()
    RED = auto()


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


def normalize_payload(payload: Dict) -> None:
    """Align recorded SGLang request bodies with router/OpenAI expectations."""
    if payload.get("min_tokens") is not None and payload["min_tokens"] < 1:
        payload.pop("min_tokens")
    for param in [
        "cache_salt",
        "bootstrap_port",
        "stop_regex",
        "routed_experts_start_len",
        "session_params",
        "encrypt_type",
        "task",
        "stream_reasoning",
        "extra_key",
        "return_routed_experts",
        "no_stop_trim",
        "bootstrap_room",
        "priority",
        "stop_token_ids",
        "disagg_prefill_dp_rank",
        "ebnf",
        "return_hidden_states",
        "continue_final_message",
        "routed_dp_rank",
        "rid",
        "custom_logit_processor",
        "data_parallel_rank",
        "use_audio_in_video",
        "lora_path",
        "separate_reasoning",
        "min_dynamic_patch",
        "max_dynamic_patch",
        "regex",
        "bootstrap_host",
        "custom_params",
        "return_cached_tokens_details",
    ]:
        if param in payload:
            payload.pop(param)

    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return
    # SGLang logs use schema_; router deserializer requires schema (OpenAI shape).
    if "schema" not in json_schema and "schema_" in json_schema:
        json_schema["schema"] = json_schema.pop("schema_")


def info_print(headers, payload, url):
    print("=" * 80)
    print(f"Sending to {url=}")
    print(f"headers={json.dumps(headers, indent=2, ensure_ascii=False)}")
    print(f"payload={json.dumps(payload, indent=2, ensure_ascii=False)}")


def read_json(filepath: str):
    if not Path(filepath).exists():
        raise ValueError(f"{filepath} doesn't exist!")
    with open(filepath, mode="r", encoding="utf-8") as f:
        return json.load(f)


def color_print(text: str, color: Color):
    RESET_CODE = "\033[0m"
    COLOR_TO_ANSI = {
        Color.LIGHT_CYAN: "\033[96m",
        Color.LIGHT_GREEN: "\033[92m",
        Color.LIGHT_YELLOW: "\033[93m",
        Color.RED: "\033[41m",
    }

    try:
        color_code = COLOR_TO_ANSI[color]
    except KeyError:
        raise NotImplementedError(f"{color} is not supported yet.")

    print(f"{color_code}{text}{RESET_CODE}", end="", flush=True)


def get_first_value(data, keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def extract_response_metrics(response):
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    else:
        data = response
    if not isinstance(data, dict):
        return {}

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    meta_info = data.get("meta_info") if isinstance(data.get("meta_info"), dict) else {}
    prompt_tokens_details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else {}
    )

    def first(keys, sources=(usage, meta_info, data)):
        for source in sources:
            value = get_first_value(source, keys)
            if value is not None:
                return value
        return None

    return {
        "prompt_tokens": first(["prompt_tokens", "input_tokens"]),
        "completion_tokens": first(["completion_tokens", "output_tokens"]),
        "reasoning_tokens": first(["reasoning_tokens"]),
        "cached_tokens": first(
            ["cached_tokens", "cached_token", "cached_prompt_tokens"],
            (prompt_tokens_details, meta_info, data),
        ),
    }


def update_metrics(metrics, new_metrics):
    for key, value in new_metrics.items():
        if value is not None:
            metrics[key] = value


def print_metrics(
    start_time, end_time, first_token_time=None, completion_tokens=None, metrics=None
):
    e2e = end_time - start_time
    ttft = first_token_time - start_time if first_token_time is not None else None
    e2e_ms = e2e * 1000
    ttft_ms = ttft * 1000 if ttft is not None else None

    metrics = metrics or {}
    completion_tokens = metrics.get("completion_tokens") or completion_tokens

    token_per_sec = None
    if completion_tokens and e2e > 0:
        token_per_sec = completion_tokens / e2e

    itl_mean = None
    if completion_tokens and completion_tokens > 1 and ttft_ms is not None:
        itl_mean = (e2e_ms - ttft_ms) / (completion_tokens - 1)

    def fmt(value, suffix=""):
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)):
            return f"{value:.4f}{suffix}"
        return str(value)

    def fmt_count(value):
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return str(value)

    rows = [
        ("ttft_ms", fmt(ttft_ms)),
        ("e2e_ms", fmt(e2e_ms)),
        ("itl_mean_ms", fmt(itl_mean)),
        ("token/s", fmt(token_per_sec, "")),
        ("prompt_tokens", fmt_count(metrics.get("prompt_tokens"))),
        ("completion_tokens", fmt_count(completion_tokens)),
        ("reasoning_tokens", fmt_count(metrics.get("reasoning_tokens"))),
        ("cached_token", fmt_count(metrics.get("cached_tokens"))),
    ]
    metric_width = max(len("metric"), *(len(metric) for metric, _ in rows))
    value_width = max(len("value"), *(len(value) for _, value in rows))
    border = f"+{'-' * (metric_width + 2)}+{'-' * (value_width + 2)}+"

    print("\n" + "=" * 80)
    print("Metrics")
    print(border)
    print(f"| {'metric'.ljust(metric_width)} | {'value'.ljust(value_width)} |")
    print(border)
    for metric, value in rows:
        print(f"| {metric.ljust(metric_width)} | {value.rjust(value_width)} |")
    print(border)


def http_request(args):
    url = f"{args.base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"}

    # construct payload
    payload = {}

    if not args.msg_path and not args.input_ids_path and not args.payload_path:
        # if there is no msg path, input_ids path, payload path
        payload["messages"] = [{"role": "user", "content": args.user_content}]
        if args.ebnf:
            payload["ebnf"] = ebnf_content
        elif args.json_schema_response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_response_format,
            }
        elif args.tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
    else:
        if args.msg_path:
            # read message path
            payload["messages"] = read_json(args.msg_path)
        elif args.payload_path:
            # read payload path
            payload = read_json(args.payload_path)
        elif args.input_ids_path:
            # read input_ids path
            input_ids = read_json(args.input_ids_path)

            from transformers import AutoTokenizer

            assert (
                args.tokenizer_path
            ), f"tokenizer_path must be provided in the input-ids-path, use --tokenizer-path <TOKENIZER PATH>"
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            prompt = tokenizer.decode(input_ids)
            print(f"{prompt=}")

            # use v1/completions api since it is already applied chat template
            payload["prompt"] = prompt
            url = f"{args.base_url}/v1/completions"

    # special field setting
    if args.model:
        payload["model"] = args.model
    if args.disable_separate_reasoning:
        payload["separate_reasoning"] = False
    if args.enable_thinking:
        # for compatibility of different platforms
        payload["chat_template_kwargs"] = {
            "thinking": True,
            "enable_thinking": True,
            "enforce_think": True,
        }
    elif args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    payload["log_metrics"] = True

    if not args.disable_stream:
        payload["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    info_print(headers, payload, url)
    if args.disable_stream:
        payload["stream"] = False
        start_time = time.perf_counter()
        res = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        end_time = time.perf_counter()
        try:
            res.raise_for_status()
            response_json = res.json()
            color_print(
                json.dumps(response_json, indent=2, ensure_ascii=False),
                Color.LIGHT_GREEN,
            )
            if args.print_response:
                print("\nParsed response:")
                print(json.dumps(response_json, indent=2, ensure_ascii=False))
            response_metrics = extract_response_metrics(response_json)
            print_metrics(
                start_time,
                end_time,
                completion_tokens=response_metrics.get("completion_tokens"),
                metrics=response_metrics,
            )
        except Exception as e:
            color_print(
                f"Request Error, Status Code={res.status_code}, Reason: {res.text} Error: {e}",
                Color.RED,
            )
            print_metrics(start_time, end_time)
    else:
        payload["stream"] = True
        normalize_payload(payload)
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = None
        metrics = {}
        last_metric_chunk = None
        res = requests.post(
            url=url,
            headers=headers,
            json=payload,
            stream=True,
        )
        try:
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode("utf-8")

                if args.raw:
                    print(decoded_line)

                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str.strip() == "[DONE]":
                        if not args.raw:
                            print("\n[DONE]")
                        break

                    now = time.perf_counter()

                    try:
                        chunk = json.loads(data_str)
                        chunk_metrics = extract_response_metrics(chunk)
                        update_metrics(metrics, chunk_metrics)
                        if any(value is not None for value in chunk_metrics.values()):
                            last_metric_chunk = chunk
                        chunk_completion_tokens = chunk_metrics.get("completion_tokens")
                        if chunk_completion_tokens is not None:
                            completion_tokens = chunk_completion_tokens

                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if reasoning_content := delta.get("reasoning_content", ""):
                                if first_token_time is None:
                                    first_token_time = now
                                if not args.raw:
                                    color_print(reasoning_content, Color.LIGHT_CYAN)

                            if content := delta.get("content", ""):
                                if first_token_time is None:
                                    first_token_time = now
                                if not args.raw:
                                    color_print(content, Color.LIGHT_GREEN)

                            if not args.raw and (
                                tool_calls := delta.get("tool_calls", "")
                            ):
                                tc = tool_calls[0]
                                if func_name := tc.get("function", {}).get("name"):
                                    color_print(
                                        f"\n\n[Tool Call Detected]: Function={func_name}\nArgument:",
                                        Color.LIGHT_YELLOW,
                                    )
                                if func_arg := tc.get("function", {}).get("arguments"):
                                    color_print(func_arg, Color.LIGHT_YELLOW)

                    except json.JSONDecodeError:
                        continue
            end_time = time.perf_counter()
            if args.print_response and last_metric_chunk is not None:
                print("\nLast metric chunk:")
                print(json.dumps(last_metric_chunk, indent=2, ensure_ascii=False))
            print_metrics(
                start_time,
                end_time,
                first_token_time=first_token_time,
                completion_tokens=completion_tokens,
                metrics=metrics,
            )

        except Exception as e:
            end_time = time.perf_counter()
            color_print(
                f"Request Error, Status Code={res.status_code}, Reason: {res.text} Error: {e}",
                Color.RED,
            )
            print_metrics(
                start_time,
                end_time,
                first_token_time=first_token_time,
                completion_tokens=completion_tokens,
                metrics=metrics,
            )


def openai_request(args):
    client = openai.OpenAI(
        base_url=f"{args.base_url}/v1/chat/completions",
        api_key=args.api_key,
    )
    model_id = list(client.models.list())[0].id

    extra_body = {}
    if args.ebnf:
        extra_body["ebnf"] = ebnf_content

    if args.disable_stream:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "who are you"}],
        )
        end_time = time.perf_counter()
        print(response)
        response_metrics = extract_response_metrics(response)
        print_metrics(
            start_time,
            end_time,
            completion_tokens=response_metrics.get("completion_tokens"),
            metrics=response_metrics,
        )
    else:
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = None
        metrics = {}
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
            stream_options={"include_usage": True},
        )
        for chunk in response_stream:
            now = time.perf_counter()
            chunk_metrics = extract_response_metrics(chunk)
            update_metrics(metrics, chunk_metrics)
            chunk_completion_tokens = chunk_metrics.get("completion_tokens")
            if chunk_completion_tokens is not None:
                completion_tokens = chunk_completion_tokens

            choices = chunk.choices
            if choices:
                choice = choices[0]
                if reasoning_content := getattr(
                    choice.delta, "reasoning_content", None
                ):
                    if first_token_time is None:
                        first_token_time = now
                    print(reasoning_content, end="", flush=True)
                if content := choice.delta.content:
                    if first_token_time is None:
                        first_token_time = now
                    print(content, end="", flush=True)
                if tool_calls := choice.delta.tool_calls:
                    print(tool_calls[0], flush=True)
        end_time = time.perf_counter()
        print_metrics(
            start_time,
            end_time,
            first_token_time=first_token_time,
            completion_tokens=completion_tokens,
            metrics=metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--base-url", type=str, default="http://localhost:8888")
    parser.add_argument("--api-key", type=str, default="JustKeepMe")
    parser.add_argument(
        "--model", type=str, help="override the model field in the payload"
    )

    parser.add_argument("--disable-stream", action="store_true")
    parser.add_argument(
        "--backend", type=str, default="http", choices=["http", "openai"]
    )
    parser.add_argument("--user-content", type=str, default="Who are you")
    parser.add_argument("--tokenizer-path", type=str, help="The path of tokenizer path")

    # extra kwargs in the payload
    think_mutex_group = parser.add_mutually_exclusive_group()
    think_mutex_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Whether to enable reasoning",
    )
    think_mutex_group.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Whether to disable reasoning",
    )

    parser.add_argument(
        "--disable-separate-reasoning",
        action="store_true",
        help="Whether to separate reasoning",
    )

    parser.add_argument(
        "--raw", action="store_true", help="Whether to print raw sse content"
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print parsed response or final metric chunk for debugging",
    )

    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--ebnf", action="store_true", help="Constrained Decoding for EBNF format"
    )
    mutex_group.add_argument(
        "--json-schema-response-format",
        action="store_true",
        help="JSON Schema Response Format",
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
