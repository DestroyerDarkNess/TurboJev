#!/usr/bin/env python3
"""Deterministic stdlib-only load harness for the TurboJev local HTTP server.

The harness is intentionally external to the Rust runtime. It consumes a
versioned JSON plan and emits a versioned JSON report. Remote targets are
rejected by default to keep accidental public load generation out of normal
verification; pass --allow-remote explicitly when that is intended.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PLAN_CONTRACT = "turbojev.server-load-plan.v1"
REPORT_CONTRACT = "turbojev.server-load-report.v1"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip("[]").lower()
    return normalized in {"localhost", "127.0.0.1", "::1"} or normalized.startswith("127.")


def validate_plan(plan: dict, allow_remote: bool) -> tuple[str, bytes, dict[str, str], float]:
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError(f"plan contract must be {PLAN_CONTRACT}")
    base = str(plan.get("baseUrl", ""))
    route = str(plan.get("route", ""))
    if not route.startswith("/"):
        raise ValueError("route must start with /")
    if plan.get("method") != "POST":
        raise ValueError("v1 load harness currently requires method=POST")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("baseUrl must be an absolute http(s) URL")
    if not allow_remote and not is_loopback_host(parsed.hostname):
        raise ValueError("remote load targets require --allow-remote")
    concurrency = int(plan.get("concurrency", 0))
    total = int(plan.get("totalRequests", 0))
    if concurrency < 1 or total < 1:
        raise ValueError("concurrency and totalRequests must be positive")
    if concurrency > 1024:
        raise ValueError("concurrency exceeds hard harness limit 1024")
    timeout_ms = int(plan.get("timeoutMs", 30_000))
    if timeout_ms < 1:
        raise ValueError("timeoutMs must be positive")
    headers = {str(k): str(v) for k, v in dict(plan.get("headers", {})).items()}
    headers.setdefault("Content-Type", "application/json")
    body = json.dumps(plan.get("body", {}), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    target = base.rstrip("/") + route
    return target, body, headers, timeout_ms / 1000.0


def one_request(index: int, target: str, body: bytes, headers: dict[str, str], timeout: float) -> dict:
    request_headers = dict(headers)
    request_headers.setdefault("X-TurboJev-Request-Id", f"load-{index:08d}")
    request = urllib.request.Request(target, data=body, headers=request_headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = int(response.status)
            error = None
    except urllib.error.HTTPError as exc:
        exc.read()
        status = int(exc.code)
        error = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # network harness must record transport failures
        status = 0
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"status": status, "latencyMs": elapsed_ms, "error": error}


def run(plan: dict, allow_remote: bool) -> dict:
    target, body, headers, timeout = validate_plan(plan, allow_remote)
    concurrency = int(plan["concurrency"])
    total = int(plan["totalRequests"])
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="turbojev-load") as pool:
        futures = [pool.submit(one_request, i, target, body, headers, timeout) for i in range(total)]
        for future in as_completed(futures):
            results.append(future.result())
    wall_ms = (time.perf_counter() - started) * 1000.0

    latencies = [float(item["latencyMs"]) for item in results]
    status_counts = Counter(str(item["status"]) for item in results)
    successes = sum(1 for item in results if 200 <= int(item["status"]) < 300)
    errors = [item["error"] for item in results if item["error"]][:32]
    completed = len(results)
    report = {
        "contract": REPORT_CONTRACT,
        "planContract": PLAN_CONTRACT,
        "target": target,
        "concurrency": concurrency,
        "totalRequests": total,
        "completedRequests": completed,
        "successes": successes,
        "failures": completed - successes,
        "statusCounts": dict(sorted(status_counts.items(), key=lambda pair: int(pair[0]))),
        "latencyMs": {
            "min": min(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "wallMs": wall_ms,
        "requestsPerSecond": (completed / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = run(plan, args.allow_remote)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({report['completedRequests']} requests, {report['requestsPerSecond']:.2f} req/s)")


if __name__ == "__main__":
    main()
