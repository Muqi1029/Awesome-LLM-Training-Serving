import json
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/root/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
    )

    parser.add_argument(
        "--num-prompt-ratio",
        type=int,
        default=10,
        help="num_prompt = num-prompt-ratio * request-rate",
    )
    parser.add_argument(
        "--num-prompt",
        type=int,
        help="if num-prompt is set, it will override num-prompt-ratio",
    )

    parser.add_argument("--random-input-len", type=int, required=True)
    parser.add_argument("--random-output-len", type=int, required=True)

    parser.add_argument("--mode", choices=["general", "slo"], required=True)

    # Args for general test
    parser.add_argument("--step", default=1, type=int)

    # Args for both general and slo test: [left, right], these are required
    parser.add_argument("--left", type=int)
    parser.add_argument("--right", type=int)

    parser.add_argument("--output-dir", type=str)

    return parser.parse_args()


def read_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_slo(request_rate, args):
    print(f"[SLO Mode] Benchmarking {request_rate}")
    output_file = f"{args.output_dir}/slo.jsonl"
    run(request_rate, args, output_file=output_file)
    data = read_jsonl(output_file)
    for item in data:
        if item["request_rate"] != request_rate:
            continue

        if (
            item["p99_ttft_ms"] > 3000
            or item["p99_tpot_ms"] > 100
            or item["p99_itl_ms"] > 100
        ):
            return False
    return True


def run(request_rate, args, output_file=None):
    if args.num_prompt is None:
        args.num_prompt = request_rate * args.num_prompt_ratio

    if output_file is None:
        output_file = f"{args.output_dir}/{datetime.now().strftime("%m%d")}_input{args.random_input_len}_output{args.random_output_len}.jsonl"

    subprocess.run(
        [
            "python",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang-oai",
            "--host",
            str(args.host),
            "--port",
            str(args.port),
            "--request-rate",
            str(request_rate),
            "--max-concurrency",
            str(request_rate),
            "--dataset-name",
            "random",
            "--dataset-path",
            args.dataset_path,
            "--num-prompt",
            str(args.num_prompt),
            "--random-input-len",
            str(args.random_input_len),
            "--random-range-ratio",
            "1",
            "--random-output-len",
            str(args.random_output_len),
            "--output-file",
            output_file,
        ],
        check=True,
    )


def warmup(args):
    run(32, args)


def test_slo():
    left, right = args.left, args.right
    while left < right:
        mid = (left + right) // 2
        is_ok = check_slo(mid, args)

        if is_ok:
            left = mid + 1
        else:
            # not satisfy
            right = mid

    # TODO: save results to some files
    print(f"The maximum concurrency satisfying SLO is {left - 1}")


def test_general(args):
    for request_rate in range(args.left, args.right + 1, args.step):
        print(f"[General Mode] Benchmarking {request_rate}")
        run(request_rate, args)


def main(args):
    warmup()
    if args.mode == "general":
        test_general(args)
    elif args.mode == "slo":
        test_slo(args)
    else:
        raise NotImplementedError(f"{args.mode} has not been supported yet")
    print("Generating all test data, now starting graph")
    # plot()


def prepare_run(args):
    if args.output_dir:
        args.output_dir = args.output_dir.strip("/ \t\n")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = "."


if __name__ == "__main__":
    args = parse_args()
    prepare_run(args)
    main(args)
