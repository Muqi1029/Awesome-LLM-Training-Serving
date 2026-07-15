# python bench_router.py --requests-path ../data
# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
import asyncio
import codecs
import json
import logging
import os
import random
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import AsyncIterator, Dict, List, Optional

import aiohttp
import numpy as np
from tqdm import tqdm

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from benchmark_utils import print_table  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB


def _create_bench_client_session(max_concurrency: int, api_key: str):
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout,
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
        connector=connector,
        headers={"Authorization": "Bearer " + api_key},
    )


@dataclass
class OutputMetric:
    payload: Dict = field(default_factory=dict)
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    success: bool = False
    out_text: str = ""
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None


def read_session_requests(requests_path: str) -> List[List[Dict]]:
    session_requests = []

    num_sessions = 0
    num_requests = 0

    for filepath in sorted(glob(f"{requests_path}/**/*.jsonl", recursive=True)):
        logger.info(f"Reading {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            num_sessions += 1
            requests = []
            for line in f:
                if not line.strip():
                    continue
                num_requests += 1
                requests.append(json.loads(line))
            session_requests.append(requests)
    logger.info(f"Totally get {num_sessions=} {num_requests=}")
    return session_requests


async def iter_sse_data(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    async for chunk_bytes in response.content.iter_any():
        buffer += decoder.decode(chunk_bytes)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[len("data:") :].lstrip()

    buffer += decoder.decode(b"", final=True)
    line = buffer.strip()
    if line and line.startswith("data:"):
        yield line[len("data:") :].lstrip()


def update_stream_output(
    output: OutputMetric,
    text: str,
    start_time: float,
) -> None:
    if not text:
        return

    output.out_text += text
    if output.ttft_ms == 0.0:
        output.ttft_ms = (time.perf_counter() - start_time) * 1000


def stream_text_or_empty(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def get_cached_tokens(usage_data: Dict) -> int:
    cached_tokens = usage_data.get("cached_tokens")
    if cached_tokens is not None:
        return int(cached_tokens)

    prompt_tokens_details = usage_data.get("prompt_tokens_details") or {}
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = prompt_tokens_details.get("cached_tokens")
        if cached_tokens is not None:
            return int(cached_tokens)

    return 0


async def request_func(
    args: Namespace,
    session: aiohttp.ClientSession,
    request_url: str,
    session_payloads: List[Dict],
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
):
    session_outputs = []
    async with sem:
        # set model and stream
        # Keep payloads in the same session strictly sequential.
        for payload_index, raw_payload in enumerate(session_payloads):
            payload = raw_payload.copy()
            if args.model:
                payload["model"] = args.model
            payload["stream"] = True
            payload["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }
            output = OutputMetric(payload=payload)
            st = time.perf_counter()
            try:
                async with session.post(url=request_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk in iter_sse_data(response):
                            if chunk == "[DONE]":
                                continue

                            data = json.loads(chunk)
                            usage_data = data.get("usage") or {}
                            if usage_data:
                                output.prompt_tokens = usage_data.get(
                                    "prompt_tokens", 0
                                )
                                output.completion_tokens = usage_data.get(
                                    "completion_tokens", 0
                                )
                                output.cached_tokens = get_cached_tokens(usage_data)

                            choices = data.get("choices") or []
                            if not choices:
                                continue

                            choice = choices[0]
                            if finish_reason := choice.get("finish_reason"):
                                output.finish_reason = finish_reason

                            delta = choice.get("delta") or {}
                            text_parts = [
                                # Reasoning models stream thoughts via
                                # `reasoning_content`; count them like content.
                                stream_text_or_empty(delta.get("reasoning_content")),
                                stream_text_or_empty(delta.get("content")),
                            ]

                            if tool_calls := delta.get("tool_calls"):
                                for tool_call in tool_calls:
                                    function = tool_call.get("function") or {}
                                    if func_name := function.get("name"):
                                        text_parts.append(
                                            "\n\n[Tool Call Detected]: "
                                            f"Function={func_name}\nArgument:"
                                        )
                                    if func_arg := function.get("arguments"):
                                        text_parts.append(
                                            stream_text_or_empty(func_arg)
                                        )

                            update_stream_output(
                                output,
                                "".join(text_parts),
                                st,
                            )

                        output.latency_ms = (time.perf_counter() - st) * 1000
                        output.success = True
                    else:
                        output.latency_ms = (time.perf_counter() - st) * 1000
                        output.error_message = await response.text()
                        print(output.error_message)
                        output.success = False
            except Exception:
                exc_info = sys.exc_info()
                error_message = "".join(traceback.format_exception(*exc_info))
                logger.error(error_message)
                output.error_message = error_message
                if st > 0.0:
                    output.latency_ms = (time.perf_counter() - st) * 1000
                output.success = False
            finally:
                session_outputs.append(output)

            if payload_index + 1 < len(session_payloads):
                await wait_for_request_interval(args.request_rate)

    if pbar:
        pbar.update(1)
    return session_outputs


def filter_outputs(outputs: List[OutputMetric]) -> List[OutputMetric]:
    filtered_outputs = []
    for output in outputs:
        if output.success and output.prompt_tokens >= 1:
            filtered_outputs.append(output)
    return filtered_outputs


def flatten_outputs(outputs):
    res = []
    for output in outputs:
        res.extend(output)
    return res


def calculate_itl_ms(output: OutputMetric) -> Optional[float]:
    if output.completion_tokens <= 1 or output.ttft_ms <= 0.0:
        return None

    generation_time_ms = output.latency_ms - output.ttft_ms
    if generation_time_ms < 0.0:
        return None
    return generation_time_ms / (output.completion_tokens - 1)


def format_mean(values: List[float], precision: int = 2) -> str:
    if not values:
        return "N/A"
    return f"{np.mean(values):.{precision}f}"


def format_percentile(
    values: List[float], percentile: float, precision: int = 2
) -> str:
    if not values:
        return "N/A"
    return f"{np.percentile(values, percentile):.{precision}f}"


def handle_outputs(
    outputs: List[List[OutputMetric]],
    duration_s: float,
    max_concurrency: int,
    request_rate: float,
    completion_tokens_output_path: Optional[str] = None,
    finish_reason_length_output_path: Optional[str] = None,
):
    # filter failed requests
    outputs = flatten_outputs(outputs)
    filtered_outputs = filter_outputs(outputs)
    num_total_requests = len(outputs)
    num_success_requests = len(filtered_outputs)
    num_failed_requests = num_total_requests - num_success_requests
    if len(filtered_outputs) != len(outputs):
        if num_failed_requests > 0:
            logger.warning(f"Failed requests: {num_failed_requests}")
    if not filtered_outputs:
        print_table(
            "Benchmark Results",
            [
                ["Metric", "Value"],
                ["Total requests", str(num_total_requests)],
                ["Successful requests", "0"],
                ["Failed requests", str(num_failed_requests)],
                ["Status", "No successful requests"],
            ],
        )
        return

    ttft_ms_list = [output.ttft_ms for output in filtered_outputs]
    itl_ms_list = [
        itl_ms
        for output in filtered_outputs
        if (itl_ms := calculate_itl_ms(output)) is not None
    ]
    latency_ms_list = [output.latency_ms for output in filtered_outputs]
    prompt_tokens_list = [output.prompt_tokens for output in filtered_outputs]
    cached_tokens_list = [output.cached_tokens for output in filtered_outputs]
    cached_tokens_ratio_list = [
        output.cached_tokens / max(output.prompt_tokens - 1, 1)
        for output in filtered_outputs
    ]
    completion_tokens_list = [output.completion_tokens for output in filtered_outputs]
    total_prompt_tokens = sum(prompt_tokens_list)
    total_cached_tokens = sum(cached_tokens_list)
    # Match the per-request cache ratio denominator: prompt_tokens - 1.
    total_cacheable_prompt_tokens = total_prompt_tokens - num_success_requests
    global_cache_ratio = (
        total_cached_tokens / total_cacheable_prompt_tokens
        if total_cacheable_prompt_tokens > 0
        else 0.0
    )

    duration_s = max(duration_s, 1e-9)
    finished_requests_per_second = num_success_requests / duration_s
    output_throughput = sum(completion_tokens_list) / duration_s
    request_rate_display = (
        "unlimited" if request_rate == float("inf") else f"{request_rate:g} req/s"
    )
    print_table(
        "Benchmark Summary",
        [
            ["Metric", "Value"],
            ["Total requests", str(num_total_requests)],
            ["Successful requests", str(num_success_requests)],
            ["Failed requests", str(num_failed_requests)],
            ["Max concurrency", str(max_concurrency)],
            ["Request rate", request_rate_display],
            ["Duration", f"{duration_s:.2f} s"],
            [
                "Mean finished requests per second",
                f"{finished_requests_per_second:.2f} req/s",
            ],
            ["Output throughput", f"{output_throughput:.2f} tokens/s"],
            ["Total prompt tokens", f"{total_prompt_tokens} tokens"],
            ["Total cached tokens", f"{total_cached_tokens} tokens"],
            ["Global cache ratio", f"{global_cache_ratio:.2%}"],
        ],
    )
    print()
    print_table(
        "Latency & Token Metrics",
        [
            ["Metric", "Mean", "P95", "P99", "Unit"],
            [
                "TTFT",
                format_mean(ttft_ms_list),
                format_percentile(ttft_ms_list, 95),
                format_percentile(ttft_ms_list, 99),
                "ms",
            ],
            [
                "ITL",
                format_mean(itl_ms_list),
                format_percentile(itl_ms_list, 95),
                format_percentile(itl_ms_list, 99),
                "ms",
            ],
            [
                "Latency",
                format_mean(latency_ms_list),
                format_percentile(latency_ms_list, 95),
                format_percentile(latency_ms_list, 99),
                "ms",
            ],
            [
                "Prompt tokens",
                format_mean(prompt_tokens_list),
                format_percentile(prompt_tokens_list, 95),
                format_percentile(prompt_tokens_list, 99),
                "tokens",
            ],
            [
                "Completion tokens",
                format_mean(completion_tokens_list),
                format_percentile(completion_tokens_list, 95),
                format_percentile(completion_tokens_list, 99),
                "tokens",
            ],
            [
                "Cached tokens",
                format_mean(cached_tokens_list),
                format_percentile(cached_tokens_list, 95),
                format_percentile(cached_tokens_list, 99),
                "tokens",
            ],
            [
                "Cached token ratio",
                f"{np.mean(cached_tokens_ratio_list):.2%}",
                f"{np.percentile(cached_tokens_ratio_list, 95):.2%}",
                f"{np.percentile(cached_tokens_ratio_list, 99):.2%}",
                "ratio",
            ],
        ],
    )

    finish_reasons = ("stop", "length", "tool_calls", "abort")
    finish_reason_counts = {
        finish_reason: sum(
            output.finish_reason == finish_reason for output in filtered_outputs
        )
        for finish_reason in finish_reasons
    }
    print()
    print_table(
        "Finish Reason Statistics",
        [
            ["Finish reason", "Requests", "Percentage"],
            *[
                [
                    finish_reason,
                    str(count),
                    f"{count / num_success_requests:.2%}",
                ]
                for finish_reason, count in finish_reason_counts.items()
            ],
        ],
    )

    # dump completion tokens
    if completion_tokens_output_path and completion_tokens_list:
        logger.info(
            f"Dumping {len(completion_tokens_list)} completion tokens to "
            f"{completion_tokens_output_path}"
        )
        with open(completion_tokens_output_path, mode="w", encoding="utf-8") as f:
            json.dump(completion_tokens_list, f, ensure_ascii=False, indent=2)

    # dump finish length requests
    finish_reason_length_list = [
        output for output in filtered_outputs if output.finish_reason == "length"
    ]
    if finish_reason_length_output_path and finish_reason_length_list:
        logger.info(
            f"Dumping {len(finish_reason_length_list)} finish reason 'length' to "
            f"{finish_reason_length_output_path}"
        )
        with open(finish_reason_length_output_path, mode="w", encoding="utf-8") as f:
            json.dump(
                [asdict(output) for output in finish_reason_length_list],
                f,
                ensure_ascii=False,
                indent=2,
            )


async def wait_for_request_interval(request_rate: float) -> None:
    if request_rate == float("inf"):
        return

    interval = np.random.exponential(1.0 / request_rate)
    await asyncio.sleep(interval)


async def get_request(requests, request_rate):
    for req in requests:
        yield req
        await wait_for_request_interval(request_rate)


async def run_benchmark(args):
    # read dataset
    session_requests: List[List[Dict]] = read_session_requests(args.requests_path)
    request_url = args.base_url.rstrip("/") + "/v1/chat/completions"

    if args.debug:
        args.num_sessions = 3
        logger.info(f"Debug mode: only use {args.num_sessions} session requests")

    # prune
    if args.num_sessions:
        session_requests = session_requests[: args.num_sessions]
        logger.info(f"Pruned to {args.num_sessions} sessions requests")

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if not (args.request_rate > 0):
        raise ValueError("--request-rate must be > 0")

    sem = asyncio.Semaphore(args.max_concurrency)

    async with _create_bench_client_session(
        args.max_concurrency, args.api_key
    ) as session:
        # warmup
        pbar = None
        if args.num_warmup_sessions:
            logger.info(f"Warming up {args.num_warmup_sessions} session requests")
            warmup_session_requests = session_requests[: args.num_warmup_sessions]
            pbar = tqdm(total=len(warmup_session_requests), desc="Warmup")
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        request_func(
                            args, session, request_url, session_payloads, sem, pbar
                        )
                    )
                    for session_payloads in warmup_session_requests
                ]
            )
            logger.info(f"Warming up done")

        if pbar:
            pbar.reset(total=len(session_requests[args.num_warmup_sessions :]))
            pbar.set_description("Formally running")

        tasks = []
        benchmark_start_time = time.perf_counter()
        async for session_payloads in get_request(
            session_requests[args.num_warmup_sessions :], args.request_rate
        ):
            tasks.append(
                asyncio.create_task(
                    request_func(
                        args,
                        session,
                        request_url,
                        session_payloads,
                        sem,
                        pbar,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        if pbar:
            pbar.close()
        benchmark_end_time = time.perf_counter()
        duration_s = benchmark_end_time - benchmark_start_time

    # handle outputs
    handle_outputs(
        outputs=outputs,
        duration_s=duration_s,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        completion_tokens_output_path=args.completion_tokens_output_path,
        finish_reason_length_output_path=args.finish_reason_length_output_path,
    )


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
        "--request-rate",
        default=float("inf"),
        type=float,
        help="Mean request rate in requests/s",
    )

    parser.add_argument(
        "--num-sessions",
        default=None,
        type=int,
        help="The number of requests to benchmark",
    )
    parser.add_argument(
        "--num-warmup-sessions",
        default=3,
        type=int,
        help="The number of requests to warmup",
    )

    parser.add_argument(
        "--requests-path", type=str, help="The path of requests", required=True
    )
    parser.add_argument(
        "--completion-tokens-output-path",
        type=str,
        default=None,
        help="Optional path to dump the full completion_tokens list",
    )
    parser.add_argument(
        "--finish-reason-length-output-path",
        type=str,
        default=None,
        help="Optional path to dump outputs whose finish_reason is 'length'",
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


def test_requests():
    args = parse_args()
    requests = read_session_requests(args.requests_path)
    interval = 100
    for i in range(0, len(requests), interval):
        with open(f"requests_{i}.json", "w", encoding="utf-8") as f:
            json.dump(
                requests[i : i + interval],
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
    # test_requests()
