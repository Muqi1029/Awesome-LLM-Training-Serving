# python bench_router.py --requests-path ../data
# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
import asyncio
import json
import logging
import os
import random
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import aiohttp
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def _create_bench_client_session():
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    # Define constants for timeout and buffer size for clarity and maintainability
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


@dataclass
class OutputMetric:
    ttft_ms: float = 0.0
    itl_ms_list: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    success: bool = False
    error_message: Optional[str] = None


def read_requests(requests_path: str) -> List[Dict]:
    data: Dict[str, str] = {}
    file_paths = glob(os.path.join(requests_path, "*.json"))
    logger.info(f"Reading {len(file_paths)} files from {requests_path}")
    for file_path in file_paths:
        logger.info(f"Reading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                ts, content = item
                data[ts] = content
    sorted_items = sorted(
        data.items(), key=lambda x: datetime.strptime(x[0], DATE_FORMAT)
    )
    requests = [json.loads(content) for _, content in sorted_items]
    logger.info(f"Read {len(requests)} requests")
    return requests


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


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


async def request_func(
    args: Namespace, payload: Dict, sem: asyncio.Semaphore, pbar: Optional[tqdm] = None
):
    # set model and stream
    if args.model:
        payload["model"] = args.model
    payload["stream"] = True
    payload["stream_options"] = {
        "include_usage": True,
        "continuous_usage_stats": True,
    }
    normalize_payload(payload)

    async with sem:
        async with _create_bench_client_session() as session:
            ttft_ms = 0.0
            latency_ms = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            output = OutputMetric()

            try:
                async with session.post(
                    url=args.base_url + "/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + args.api_key},
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                            latency_ms = (time.perf_counter() - st) * 1000
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)
                                choices = data.get("choices") or []
                                if not choices:
                                    continue

                                usage_data = data.get("usage") or {}
                                if usage_data:
                                    output.prompt_tokens = usage_data.get(
                                        "prompt_tokens", 0
                                    )
                                    output.completion_tokens = usage_data.get(
                                        "completion_tokens", 0
                                    )
                                    output.cached_tokens = usage_data.get(
                                        "prompt_tokens_details", {}
                                    ).get("cached_tokens", 0)

                                # Reasoning models stream thoughts via
                                # `reasoning_content`; count them like content.
                                delta = choices[0].get("delta") or {}

                                if delta.get("reasoning_content") or delta.get(
                                    "content"
                                ):
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft_ms == 0.0:
                                        ttft_ms = (timestamp - st) * 1000
                                        output.ttft_ms = ttft_ms

                                    # Decoding phase
                                    else:
                                        itl_ms = (
                                            timestamp - most_recent_timestamp
                                        ) * 1000
                                        output.itl_ms_list.append(itl_ms)

                                    most_recent_timestamp = timestamp
                        output.latency_ms = latency_ms
                        output.success = True
                    else:
                        output.error_message = await response.text()
                        print(output.error_message)
                        output.success = False
                        return output
            except Exception:
                exc_info = sys.exc_info()
                error_message = "".join(traceback.format_exception(*exc_info))
                logger.error(error_message)
                output.error_message = error_message
                output.success = False
                return output

    if pbar:
        pbar.update(1)
    return output


def filter_outputs(outputs: List[OutputMetric]) -> List[OutputMetric]:
    filtered_outputs = []
    for output in outputs:
        if output.success and output.prompt_tokens >= 1:
            filtered_outputs.append(output)
    return filtered_outputs


def handle_outputs(outputs: List[OutputMetric], duration_s: float):
    # filter failed requests
    filtered_outputs = filter_outputs(outputs)
    if len(filtered_outputs) != len(outputs):
        num_failed_requests = len(outputs) - len(filtered_outputs)
        if num_failed_requests > 0:
            logger.warning(f"Failed requests: {num_failed_requests}")

    ttft_ms_list = [output.ttft_ms for output in filtered_outputs]
    latency_ms_list = [output.latency_ms for output in filtered_outputs]
    prompt_tokens_list = [output.prompt_tokens for output in filtered_outputs]
    cached_tokens_list = [output.cached_tokens for output in filtered_outputs]
    cached_tokens_ratio_list = [
        output.cached_tokens / max(output.prompt_tokens - 1, 1)
        for output in filtered_outputs
    ]
    completion_tokens_list = [output.completion_tokens for output in filtered_outputs]

    print(" Benchmark results ".center(100, "="))
    print(f"Output throughput: {sum(completion_tokens_list) / duration_s:.2f} tokens/s")
    print(f"Mean ttft: {np.mean(ttft_ms_list):.2f} ms")
    print(f"Mean latency: {np.mean(latency_ms_list):.2f} ms")
    print(f"Mean cached tokens: {np.mean(cached_tokens_list):.2f} tokens")
    print(f"Mean prompt tokens: {np.mean(prompt_tokens_list):.2f} tokens")
    print(f"Mean cached tokens ratio: {np.mean(cached_tokens_ratio_list):.2%}")
    print(f"Mean completion tokens: {np.mean(completion_tokens_list):.2f} tokens")
    print("=" * 100)


async def get_request(requests, request_rate):
    for req in requests:
        yield req

        if request_rate == float("inf"):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


async def run_benchmark(args):
    # read dataset
    requests = read_requests(args.requests_path)

    if args.debug:
        args.num_requests = 10
        args.warmup_requests = 3
        logger.info(f"Debug mode: only use {args.num_requests} benchmark requests")
        logger.info(f"Debug mode: only use {args.warmup_requests} warmup requests")

    # prune
    if args.num_requests:
        requests = requests[: args.num_requests]
        logger.info(f"Pruned to {len(requests)} requests")

    sem = asyncio.Semaphore(args.max_concurrency)

    # warmup
    if args.warmup_requests:
        logger.info(f"Warming up {args.warmup_requests} requests")
        warmup_requests = requests[: args.warmup_requests]
        pbar = tqdm(total=len(warmup_requests), desc="Warmup")
        await asyncio.gather(
            *[
                asyncio.create_task(request_func(args, req, sem, pbar))
                for req in warmup_requests
            ]
        )
        logger.info(f"Warming up done")

    pbar.reset(total=len(requests[args.warmup_requests :]))
    pbar.set_description("Formally running")
    tasks = []
    benchmark_start_time = time.perf_counter()
    async for req in get_request(requests[args.warmup_requests :], args.request_rate):
        tasks.append(asyncio.create_task(request_func(args, req, sem, pbar)))
    outputs = await asyncio.gather(*tasks)
    pbar.close()
    benchmark_end_time = time.perf_counter()
    duration_s = benchmark_end_time - benchmark_start_time

    # handle outputs
    handle_outputs(outputs, duration_s)


def parse_args():
    parser = ArgumentParser(description="Benchmark router")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8888", help="The base URL of the router"
    )
    parser.add_argument(
        "--api-key", default="JustKeepMe", help="The API key of the router"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The model to benchmark"
    )

    parser.add_argument("--seed", type=int, default=42, help="The seed for random")

    parser.add_argument(
        "--max-concurrency", default=32, type=int, help="The max concurrency"
    )
    parser.add_argument(
        "--request-rate", default=float("inf"), type=float, help="Request rate"
    )

    parser.add_argument(
        "--num-requests",
        default=None,
        type=int,
        help="The number of requests to benchmark",
    )
    parser.add_argument(
        "--warmup-requests",
        default=100,
        type=int,
        help="The number of requests to warmup",
    )

    parser.add_argument(
        "--requests-path", type=str, help="The path of requests", required=True
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    logger.info(f"{args=}")
    set_seed(args.seed)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
