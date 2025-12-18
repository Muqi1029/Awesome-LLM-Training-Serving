"""python req.json.py --base-url http://127.0.0.1:8888 --api-key JustKeepMe --filepath <filename in the req_json>"""

import argparse
import json
import os
from pprint import pprint

import requests


def stream_req():
    res = requests.post(
        args.base_url + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
        stream=True,
    )

    print("Streaming Response:", flush=True)

    for line in res.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")

            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]  # remove "data: " prefix

                if data_str.strip() == "[DONE]":
                    print("\n[DONE]")
                    break

                try:
                    chunk = json.loads(data_str)

                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            print(content, end="", flush=True)

                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            for tc in tool_calls:
                                # get function name
                                if tc.get("function", {}).get("name"):
                                    func_name = tc["function"]["name"]
                                    call_id = tc.get("id", "unknown")
                                    print(
                                        f"\n\n[Tool Call Detected]: ID={call_id}, Function={func_name}"
                                    )
                                    print("Arguments: ", end="", flush=True)

                                # argument streaming
                                args_chunk = tc.get("function", {}).get("arguments")
                                if args_chunk:
                                    print(args_chunk, end="", flush=True)

                except json.JSONDecodeError:
                    continue


def normal_req():
    res = requests.post(
        args.base_url + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
    )
    res.raise_for_status()
    pprint(res.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", type=str, default="JustKeepMe")
    parser.add_argument("--filepath", type=str, default="badcase.json")
    args = parser.parse_args()

    # load payload
    with open(os.path.join("req_json", args.filepath), "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload["stream"]:
        stream_req()
    else:
        normal_req()
