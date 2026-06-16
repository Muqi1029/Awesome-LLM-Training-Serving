import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class RequestOutput:
    success: bool = False
    latency_ms: float = 0.0
    content_list: list = field(default_factory=list)


RequestFunc = Callable[
    [str, tqdm, asyncio.Semaphore], Coroutine[Any, Any, RequestOutput]
]


async def vllm_request_func(
    prompt: str, pbar: tqdm, sem: asyncio.Semaphore
) -> RequestOutput:
    async with sem:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "",
                "max_completion_tokens": args.output_len,
                # "stop": "<|im_end|>",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "n": args.n,
                "use_beam_search": not args.disable_beam_search,
                "early_stopping": False,
                # "temperature": 0.0,
                # "beam_width_array": [16, 16, 32, 32],
                # "repetition_penalty": 1.0,
                # "length_penalty": 1.0
            }
            st = time.perf_counter()
            output = RequestOutput()
            output.content_list = ["" for _ in range(args.n)]
            try:
                async with session.post(
                    url=args.base_url + "/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": args.api_key,
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()

                        for choice in response_json["choices"]:
                            output.content_list[choice["index"]] = choice["message"][
                                "content"
                            ]
                            if choice["index"] == 0:
                                # use the first choice as latency metric
                                output.latency_ms = (time.perf_counter() - st) * 1000
                        output.success = True
                    else:
                        print(response.reason)
            except Exception:
                exc_info = sys.exc_info()
                print("".join(traceback.format_exception(*exc_info)))
            pbar.update(1)
            return output


async def sgl_request_func(
    prompt: str, pbar: tqdm, sem: asyncio.Semaphore
) -> RequestOutput:
    async with sem:
        async with aiohttp.ClientSession() as session:
            payload = {
                "max_completion_tokens": args.output_len,
                "stop": "<|im_end|>",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "n": args.n,
                "use_beam_search": True,
                "num_beam_samples": 3,
                "stream": False,
            }
            st = time.perf_counter()
            output = RequestOutput()
            output.content_list = ["" for _ in range(args.n * 3)]
            try:
                async with session.post(
                    url=args.base_url + "/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": args.api_key,
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()

                        for choice in response_json["choices"]:
                            output.content_list[choice["index"]] = choice["message"][
                                "content"
                            ]
                            if choice["index"] == 0:
                                # use the first choice as latency metric
                                output.latency_ms = (time.perf_counter() - st) * 1000
                        output.success = True
                    else:
                        print(response.reason)
            except Exception:
                exc_info = sys.exc_info()
                print("".join(traceback.format_exception(*exc_info)))
            pbar.update(1)
            return output


dispatcher: dict[str, RequestFunc] = {
    "vllm": vllm_request_func,
    "trtllm": vllm_request_func,
    "sgl": sgl_request_func,
}


def compute_metric(values: list, name: str):
    mean_ms = np.mean(values or 0)
    median_ms = np.median(values or 0)
    std_ms = np.std(values or 0)
    p99_ms = np.percentile(values or 0, 99)
    print(f"{name=:<15} {mean_ms=:.2f} {median_ms=:.2f} {std_ms=:.2f} {p99_ms=:.2f}")


def sample_input_lens(input_len: int, input_ratio: float, num: int) -> list[int]:
    if input_ratio >= 1.0:
        return [input_len] * num
    low = max(int(input_len * input_ratio), 1)
    return [random.randint(low, input_len) for _ in range(num)]


def fake_prompt(input_len: int, tokenizer) -> str:
    system_len = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
    )
    user_prompt_len = input_len - system_len

    token_id = tokenizer.encode("who", add_special_tokens=False)[0]
    token_ids = [token_id] * user_prompt_len

    # token_ids = random.sample(range(1, 1 + tokenizer.vocab_size), k=user_prompt_len)
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return prompt


async def main(args):
    pbar = tqdm(total=args.num_requests)
    random.seed(args.seed)

    request_func = dispatcher[args.backend]

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            item = json.load(f)
            prompts = [item["prompt"]] * args.num_requests
    elif args.input_len:
        assert args.tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        input_lens = sample_input_lens(
            args.input_len, args.input_ratio, args.num_requests
        )
        print(
            f"input_lens: min={min(input_lens)}, max={max(input_lens)}, "
            f"mean={np.mean(input_lens):.1f}"
        )
        prompts = [fake_prompt(length, tokenizer) for length in input_lens]
    else:
        raise NotImplementedError("Has not supported None input_file")

    # exit(1)
    sem = asyncio.Semaphore(args.max_concurrency)
    outputs = await asyncio.gather(
        *[
            asyncio.create_task(request_func(prompts[i], pbar, sem))
            for i in range(args.num_requests)
        ]
    )

    with open(
        os.path.join(args.output_dir, f"{args.backend}_output.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump([asdict(item) for item in outputs], f, ensure_ascii=False, indent=2)

    # filter
    filtered_output = [item for item in outputs if item.success]
    if len(filtered_output) != len(outputs):
        print("WARNING: some requests are not finished")

    latencies = [item.latency_ms for item in outputs]
    compute_metric(latencies, "latentcy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--api-key", default="JustKeepMe")

    parser.add_argument("--seed", type=int, default=42, help="")

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        help="Used for tokenizer to fake a `input-len` request",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="The output directory that saves bench result",
    )

    parser.add_argument("--backend", choices=dispatcher.keys(), required=True)

    parser.add_argument(
        "--beam-width",
        "--n",
        dest="n",
        default=32,
        type=int,
        help="Beam width or parallel sampling numbers",
    )

    parser.add_argument("--max-concurrency", default=32, type=int)
    parser.add_argument(
        "--num-requests",
        default=1000,
        type=int,
        help="The number of request used for test",
    )
    parser.add_argument("--input-len", type=int, help="Input len set for each request")
    parser.add_argument(
        "--input-ratio",
        type=float,
        default=1.0,
        help="Per-request input length sampled uniformly in "
        "[input_len * ratio, input_len]. Default 1.0 uses exact input_len.",
    )
    parser.add_argument(
        "--output-len", default=2, type=int, help="Output len set for each request"
    )

    parser.add_argument(
        "--input-file", default=None, type=str, help="Input file for request"
    )

    parser.add_argument(
        "--disable-beam-search",
        action="store_true",
        default=False,
        help="Disable beam search to benchmark",
    )

    args = parser.parse_args()
    print(f"{args=}")

    asyncio.run(main(args))
