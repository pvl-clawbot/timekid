"""Microbenchmarks for timekid overhead.

Goals:
- Compare overhead of Timer context-manager path vs FastTimer.
- Compare FastTimer pure-python fallback vs native (Rust) backend when available.
- Compare FastTimer keying with int ids (key_id) vs raw string keys.

These are *overhead* benchmarks: the work inside the timed region is intentionally tiny.

Run locally:

  PYTHONPATH=src python benchmarks/bench_overhead.py

JSON output (for CI + tracking):

  PYTHONPATH=src python benchmarks/bench_overhead.py --json

If the native extension is built/installed, the script will detect it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _empty() -> None:
    return None


def _measure(fn, n: int) -> float:
    """Return average nanoseconds per call over n iterations."""
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n


def _fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.2f} µs"
    return f"{ns/1_000_000:.2f} ms"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    p.add_argument("--n", type=int, default=200_000, help="Iterations for FastTimer/baseline")
    args = p.parse_args(argv)

    # Ensure we can import from src when run from repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from timekid.timer import Timer
    import timekid.fast as fast

    has_native = fast.FastTimer.__module__ == "timekid._fast"

    n_fast = int(args.n)
    n_timer = max(10_000, n_fast // 10)  # slower, fewer iterations

    # Baseline: empty Python function call overhead.
    baseline = _measure(_empty, n_fast)

    # Timer context manager overhead.
    timer = Timer()

    def timer_ctx():
        with timer["x"]:
            pass

    timer_ns = _measure(timer_ctx, n_timer)

    # FastTimer: int key id
    ft_int = fast.FastTimer()
    key_int = ft_int.key_id("x")

    def fast_int():
        tok = ft_int.start(key_int)
        ft_int.stop(tok)

    fast_int_ns = _measure(fast_int, n_fast)

    # FastTimer: string key
    ft_str = fast.FastTimer()
    key_str = "x"

    def fast_str():
        tok = ft_str.start(key_str)
        ft_str.stop(tok)

    fast_str_ns = _measure(fast_str, n_fast)

    payload = {
        "python": sys.version.split()[0],
        "backend": "native" if has_native else "python",
        "backend_module": fast.FastTimer.__module__,
        "iterations": {"fast": n_fast, "timer": n_timer},
        "ns_per_call": {
            "baseline_empty_call": baseline,
            "timer_context_manager": timer_ns,
            "fasttimer_int_key": fast_int_ns,
            "fasttimer_str_key": fast_str_ns,
        },
        "ratios": {
            "fasttimer_int_over_baseline": fast_int_ns / baseline,
            "fasttimer_str_over_baseline": fast_str_ns / baseline,
            "timerctx_over_baseline": timer_ns / baseline,
            "timerctx_over_fast_int": timer_ns / fast_int_ns if fast_int_ns else None,
            "fast_str_over_fast_int": fast_str_ns / fast_int_ns if fast_int_ns else None,
        },
    }

    if args.json:
        print(json.dumps(payload, sort_keys=True))
        return 0

    print("timekid overhead microbench")
    print(f"Python: {payload['python']}")
    print(f"FastTimer backend: {payload['backend']} ({payload['backend_module']})")
    print(f"Iterations: fast={n_fast:,} timer_ctx={n_timer:,}")
    print()
    print(f"baseline empty call: {_fmt(baseline)}")
    print(f"Timer context manager: {_fmt(timer_ns)}")
    print(f"FastTimer start/stop (int key): {_fmt(fast_int_ns)}")
    print(f"FastTimer start/stop (str key): {_fmt(fast_str_ns)}")
    print()
    print(f"Fast(str)/Fast(int): {payload['ratios']['fast_str_over_fast_int']:.2f}x")
    print(f"TimerCtx/Fast(int): {payload['ratios']['timerctx_over_fast_int']:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
