import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class RequestOutput:
    success: bool = False
    latency_ms: float = 0.0
    content_list: list = field(default_factory=list)


async def vllm_request_func(prompt: str, pbar: tqdm, sem: asyncio.Semaphore):
    async with sem:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "",
                "max_completion_tokens": args.output_len,
                # "stop": "<|im_end|>",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                # "temperature": 0.0,
                "n": args.n,
                # "best_of": 8,
                "use_beam_search": not args.disable_beam_search,
                # "stream": True,
                "early_stopping": False,
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


async def sgl_request_func(prompt: str, pbar: tqdm, sem: asyncio.Semaphore):
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


dispatcher = {
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


def fake_prompt(input_len: int, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    system_len = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
    )
    user_prompt_len = input_len - system_len

    token_id = tokenizer.encode("hello", add_special_tokens=False)[0]
    token_ids = [token_id] * user_prompt_len

    # token_ids = random.sample(range(1, 1 + tokenizer.vocab_size), k=user_prompt_len)
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(f"{prompt=}")

    adjusted_input_len = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
    )
    print(f"{adjusted_input_len=}")
    return prompt


async def main(args):
    pbar = tqdm(total=args.num_requests)
    random.seed(args.seed)

    request_func = dispatcher[args.backend]

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            item = json.load(f)
            prompt = item["prompt"]
    elif args.input_len:
        assert args.tokenizer_path
        prompt = fake_prompt(args.input_len, args.tokenizer_path)
    else:
        raise NotImplementedError("Has not supported None input_file")

    # exit(1)
    sem = asyncio.Semaphore(args.max_concurrency)
    outputs = await asyncio.gather(
        *[
            asyncio.create_task(request_func(prompt, pbar, sem))
            for _ in range(args.num_requests)
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
