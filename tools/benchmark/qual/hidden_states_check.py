#!/usr/bin/env python3
"""Compare hidden states / logprobs across multiple serving endpoints."""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

PROMPT = "I am Muqi Li, Who are you?"
PAYLOAD = {
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "max_tokens": 10,
    "ignore_eos": True,
    "logprobs": True,
    "top_logprobs": 1,
    "return_hidden_states": True,
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-urls",
        nargs="+",
        required=True,
        help="One or more OpenAI-compatible base URLs, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for hidden-state allclose check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for hidden-state allclose check",
    )
    return parser.parse_args()


async def fetch_completion(
    session: aiohttp.ClientSession, base_url: str
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with session.post(url, json=PAYLOAD) as response:
        body = await response.json()
        if response.status >= 400:
            raise RuntimeError(
                f"{base_url} returned HTTP {response.status}: "
                f"{json.dumps(body, ensure_ascii=False)[:500]}"
            )
        print(json.dumps(body, indent=4, ensure_ascii=False))
        return body


def extract_text(resp: Dict[str, Any]) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def extract_hidden_states(resp: Dict[str, Any]) -> Optional[np.ndarray]:
    choice = resp["choices"][0]
    hidden = choice.get("hidden_states")
    if hidden is None:
        # some builds put it under message
        hidden = choice.get("message", {}).get("hidden_states")
    if hidden is None or len(hidden) == 0:
        return None
    return np.asarray(hidden, dtype=np.float64)


def extract_token_logprobs(resp: Dict[str, Any]) -> Optional[np.ndarray]:
    logprobs = resp["choices"][0].get("logprobs")
    if not logprobs:
        return None
    content = logprobs.get("content") or []
    values = [item["logprob"] for item in content if "logprob" in item]
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def summarize_pair(
    name: str,
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    rtol: float,
    atol: float,
) -> Tuple[bool, Dict[str, Any]]:
    if a is None or b is None:
        return False, {"error": f"missing {name} on one or both endpoints"}
    if a.shape != b.shape:
        return False, {
            "error": f"shape mismatch: {a.shape} vs {b.shape}",
        }
    diff = np.abs(a - b)
    cos = float(
        np.dot(a.ravel(), b.ravel())
        / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    )
    ok = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    return ok, {
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cos,
        "allclose": ok,
    }


async def main() -> int:
    args = parse_args()
    if len(args.base_urls) < 1:
        print("Need at least one --base-urls", file=sys.stderr)
        return 1

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        responses = await asyncio.gather(
            *[fetch_completion(session, url) for url in args.base_urls]
        )

    results: List[Dict[str, Any]] = []
    for url, resp in zip(args.base_urls, responses):
        hs = extract_hidden_states(resp)
        lp = extract_token_logprobs(resp)
        item = {
            "base_url": url,
            "text": extract_text(resp),
            "hidden_states_shape": list(hs.shape) if hs is not None else None,
            "num_token_logprobs": int(lp.shape[0]) if lp is not None else None,
        }
        results.append(item)
        print(f"=== {url} ===")
        print(f"text: {item['text']!r}")
        print(f"hidden_states shape: {item['hidden_states_shape']}")
        print(f"token logprobs: {item['num_token_logprobs']}")
        print()

    if len(args.base_urls) == 1:
        # single endpoint: dump raw response for inspection
        print(json.dumps(responses[0], indent=2, ensure_ascii=False)[:4000])
        return 0 if extract_hidden_states(responses[0]) is not None else 1

    ref_url = args.base_urls[0]
    ref_hs = extract_hidden_states(responses[0])
    ref_lp = extract_token_logprobs(responses[0])
    all_ok = True

    print(f"=== compare vs {ref_url} ===")
    for url, resp in zip(args.base_urls[1:], responses[1:]):
        hs_ok, hs_stats = summarize_pair(
            "hidden_states",
            ref_hs,
            extract_hidden_states(resp),
            args.rtol,
            args.atol,
        )
        lp_ok, lp_stats = summarize_pair(
            "logprobs",
            ref_lp,
            extract_token_logprobs(resp),
            args.rtol,
            args.atol,
        )
        text_match = extract_text(responses[0]) == extract_text(resp)
        print(f"--- {url} ---")
        print(f"text_match: {text_match}")
        print(f"hidden_states: {json.dumps(hs_stats, indent=2)}")
        print(f"logprobs: {json.dumps(lp_stats, indent=2)}")
        all_ok = all_ok and hs_ok and text_match

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
