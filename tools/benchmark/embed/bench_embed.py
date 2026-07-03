import asyncio
import json
import logging
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
from tqdm import tqdm

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


@dataclass
class OutputMetric:
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0
    success: bool = False
    failed_reason: str = ""
    response_content: Any = None


@dataclass
class BenchmarkResult:
    target_qps: float
    actual_qps: float
    elapsed_s: float
    outputs: list[OutputMetric]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:9091",
        type=str,
        help="Base URL of server endpoint",
    )
    parser.add_argument(
        "--api-key", default="", type=str, help="API Key for the server endpoint"
    )
    parser.add_argument(
        "--qps", default=float("inf"), type=float, help="Request Per Second"
    )
    parser.add_argument(
        "--max-concurrency",
        default=128,
        type=int,
        help="Maximum number of in-flight requests",
    )
    parser.add_argument(
        "--num-requests",
        default=1000,
        type=int,
        help="The number of requests in total for this benchmark",
    )
    parser.add_argument(
        "--warmup-requests",
        default=0,
        type=int,
        help="The number of warmup requests before benchmark, not counted in metrics",
    )
    parser.add_argument("--data-path", required=True, type=str, help="The path to data")
    parser.add_argument(
        "--api", default="/embed_lexical", type=str, help="The testing API"
    )

    parser.add_argument(
        "--debug", action="store_true", help="The debug mode for testing connection"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def validate_args(args: Namespace) -> None:
    if args.qps <= 0:
        raise ValueError("qps must be positive")
    if args.num_requests < 0:
        raise ValueError("num-requests must be greater than or equal to 0")
    if args.warmup < 0:
        raise ValueError("warmup must be greater than or equal to 0")
    if args.max_concurrency <= 0:
        raise ValueError("max-concurrency must be positive")


def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        item = json.load(f)
    if not isinstance(item, list) or not item:
        raise ValueError(
            "data-path must contain a non-empty JSON list of request payloads"
        )
    return item


async def payload_generator(qps, request_data, num_requests):
    request_data_length = len(request_data)
    interval_s = 0 if qps == float("inf") else 1.0 / qps
    next_send_time = time.perf_counter()

    for send_length in range(num_requests):
        offset = send_length % request_data_length
        yield request_data[offset]
        if interval_s <= 0:
            continue
        next_send_time += interval_s
        sleep_s = next_send_time - time.perf_counter()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)


async def request_func(
    session: aiohttp.ClientSession,
    payload: dict[str, Any],
    request_endpoint: str,
    headers: dict[str, str],
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
):
    output_metric = OutputMetric(payload=payload)

    async with sem:
        st = time.perf_counter()
        try:
            async with session.post(
                url=request_endpoint, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    output_metric.success = True
                    output_metric.response_content = await read_response(response)
                    output_metric.latency_ms = (time.perf_counter() - st) * 1000
                    logger.debug("response_content=%s", output_metric.response_content)
                else:
                    output_metric.failed_reason = await response.text()
                    output_metric.latency_ms = (time.perf_counter() - st) * 1000
                    logger.error(
                        "HTTP %s: %s", response.status, output_metric.failed_reason
                    )

        except Exception:
            exc_info = sys.exc_info()
            error_message = "".join(traceback.format_exception(*exc_info))
            logger.error(error_message)
            output_metric.failed_reason = error_message

    if pbar:
        pbar.update(1)
    return output_metric


async def read_response(response: aiohttp.ClientResponse) -> Any:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        return await response.json(content_type=None)
    return await response.text()


async def run_requests(
    session: aiohttp.ClientSession,
    request_data: list[dict],
    request_endpoint: str,
    request_headers: dict[str, str],
    max_concurrency: int,
    qps: float,
    num_requests: int,
    desc: Optional[str] = None,
) -> list[OutputMetric]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = []
    pbar = tqdm(total=num_requests, desc=desc) if desc else None

    async for payload in payload_generator(qps, request_data, num_requests):
        tasks.append(
            asyncio.create_task(
                request_func(
                    session,
                    payload,
                    request_endpoint,
                    headers=request_headers,
                    sem=sem,
                    pbar=pbar,
                )
            )
        )
    results: list[OutputMetric] = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()
    return results


async def benchmark(request_endpoint, args: Namespace):

    request_data: list[dict] = read_data(args.data_path)

    request_headers = {"Content-Type": "application/json"}
    if args.api_key:
        request_headers["Authorization"] = args.api_key

    timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(limit=args.max_concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
    ) as session:
        if args.warmup > 0:
            logger.info("Start warmup with %s requests", args.warmup)
            warmup_results = await run_requests(
                session,
                request_data,
                request_endpoint,
                request_headers,
                args.max_concurrency,
                args.qps,
                args.warmup,
                desc="Warmup",
            )
            warmup_success = sum(output.success for output in warmup_results)
            logger.info(
                "Warmup finished: %s/%s succeeded",
                warmup_success,
                len(warmup_results),
            )

        benchmark_start_time = time.perf_counter()
        results = await run_requests(
            session,
            request_data,
            request_endpoint,
            request_headers,
            args.max_concurrency,
            args.qps,
            args.num_requests,
            desc="Benchmark",
        )

    benchmark_elapsed_time = time.perf_counter() - benchmark_start_time
    actual_qps = (
        len(results) / benchmark_elapsed_time if benchmark_elapsed_time > 0 else 0.0
    )

    return BenchmarkResult(args.qps, actual_qps, benchmark_elapsed_time, results)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_qps(qps: float) -> str:
    return "inf" if qps == float("inf") else f"{qps:.2f}"


def print_table(title: str, rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    width = key_width + value_width + 7
    print()
    print(title.strip().center(width, "="))
    for key, value in rows:
        print(f"| {key:<{key_width}} | {value:>{value_width}} |")
    print("=" * width)


def compute_metrics(result: BenchmarkResult):
    outputs = result.outputs
    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]

    latencies = [output.latency_ms for output in successful_outputs]
    success_rate = len(successful_outputs) / len(outputs) if outputs else 0.0

    summary_rows = [
        ("Requests", str(len(outputs))),
        ("Successful", str(len(successful_outputs))),
        ("Failed", str(len(failed_outputs))),
        ("Success rate", f"{success_rate:.2%}"),
        ("Target QPS", format_qps(result.target_qps)),
        ("Actual QPS", f"{result.actual_qps:.2f}"),
        (
            "Success QPS",
            (
                f"{len(successful_outputs) / result.elapsed_s:.2f}"
                if result.elapsed_s > 0
                else "0.00"
            ),
        ),
        ("Elapsed", f"{result.elapsed_s:.2f} s"),
    ]
    latency_rows = [
        ("Mean", f"{mean(latencies):.2f} ms"),
        ("P50", f"{percentile(latencies, 50):.2f} ms"),
        ("P90", f"{percentile(latencies, 90):.2f} ms"),
        ("P95", f"{percentile(latencies, 95):.2f} ms"),
        ("P99", f"{percentile(latencies, 99):.2f} ms"),
        ("Min", f"{min(latencies, default=0):.2f} ms"),
        ("Max", f"{max(latencies, default=0):.2f} ms"),
    ]

    print_table(" Benchmark Summary ", summary_rows)
    print_table(" Latency ", latency_rows)

    if failed_outputs:
        logger.warning(
            "First failed reason: %s", failed_outputs[0].failed_reason[:1000]
        )


async def main():
    args: Namespace = parse_args()
    logging.getLogger().setLevel(args.log_level)
    validate_args(args)
    request_endpoint = args.base_url + args.api

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

        logger.info(
            f"[DEBUG MODE]: set QPS to 3, num_requests to 10, warmup requests to 0"
        )
        args.qps = 3
        args.num_requests = 10
        args.warmup_requests = 0
    else:
        logger.info(f"[FORMAL MODE]: {args=}")

    result = await benchmark(request_endpoint, args)
    compute_metrics(result)


if __name__ == "__main__":
    asyncio.run(main())
