"""Usage: python one_request.py"""

import argparse
import json
import time
from pprint import pprint

import numpy as np
import requests

QUERY_CONTENT = "who are you"
F1_DATA_PATH = "../data/f1_demo.json"


def main(args):
    if args.mode == "query":
        content = QUERY_CONTENT
        max_tokens = args.max_tokens or 12
    elif args.mode == "f1":
        with open(F1_DATA_PATH) as f:
            content = json.load(f)["prompt"]
        max_tokens = args.max_tokens or 4
    else:
        raise ValueError(f"{args.mode} is not supported")

    payload = {
        "model": "",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": content},
        ],
        # "stop": "<|im_end|>",
        # "beam_width_array": [1, 2]
        # "num_beam_samples": 4,
        # "early_stopping": False,
        # "repetition_penalty": 0.1,
        # "length_penalty": 0.0,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
    }
    if not args.disable_beam_search:
        payload.update({"n": args.beam, "best_of": args.beam, "use_beam_search": True})

    latency_ms_list = []
    response = None
    for _ in range(args.n):
        try:
            start_tic = time.perf_counter()
            response = requests.post(
                url=f"{args.base_url}/v1/chat/completions",
                headers={
                    "Authorization": args.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            latency_ms_list.append((time.perf_counter() - start_tic) * 1000)

            pprint(response.json())
        except requests.HTTPError:
            if response:
                print(response.text)
        time.sleep(0.5)

    # remove the slowest 3
    latency_ms_list.sort()
    if len(latency_ms_list) > 3:
        latency_ms_list = latency_ms_list[3:]
        print(
            f"\033[42m {np.mean(latency_ms_list)=} in {len(latency_ms_list)} rounds \033[0m"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--api-key", default="JustKeepMe")
    parser.add_argument("--beam", type=int, default=32, help="Beam width")
    parser.add_argument("--n", type=int, default=13, help="Repeat times")
    parser.add_argument("--max-tokens", type=int, help="output len")
    parser.add_argument("--mode", type=str, choices=["query", "f1"], default="f1")
    parser.add_argument(
        "--disable-beam-search",
        action="store_true",
        help="Disable Beam Search for compatibility for other llm inference engine",
    )

    args = parser.parse_args()
    main(args)
