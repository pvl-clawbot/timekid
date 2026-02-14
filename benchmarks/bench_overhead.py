"""Microbenchmarks for timekid overhead.

Goals:
- Compare overhead of Timer context-manager path vs FastTimer.
- Compare FastTimer pure-python fallback vs native (Rust) backend when available.

These are *overhead* benchmarks: the work inside the timed region is intentionally tiny.

Run locally:

  PYTHONPATH=src python benchmarks/bench_overhead.py

If the native extension is built/installed, the script will detect it.
"""

from __future__ import annotations

import os
import sys
import time
import statistics as stats


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


def main() -> int:
    # Ensure we can import from src when run from repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from timekid.timer import Timer
    import timekid.fast as fast

    has_native = fast.FastTimer.__module__ == "timekid._fast"

    N = 2_000_00  # 200k; keeps CI runtime reasonable

    # Baseline: empty Python function call overhead.
    baseline = _measure(_empty, N)

    # Timer context manager overhead.
    timer = Timer()

    def timer_ctx():
        with timer["x"]:
            pass

    timer_ns = _measure(timer_ctx, max(10_000, N // 10))  # slower, fewer iterations

    # FastTimer start/stop overhead.
    ft = fast.FastTimer()
    key = ft.key_id("x")

    def fast_start_stop():
        tok = ft.start(key)
        ft.stop(tok)

    fast_ns = _measure(fast_start_stop, N)

    print("timekid overhead microbench")
    print(f"Python: {sys.version.split()[0]}")
    print(f"FastTimer backend: {'native' if has_native else 'python'} ({fast.FastTimer.__module__})")
    print(f"Iterations: N={N:,} (Timer ctx uses fewer)")
    print()
    print(f"baseline empty call: {_fmt(baseline)}")
    print(f"Timer context manager: {_fmt(timer_ns)}")
    print(f"FastTimer start/stop: {_fmt(fast_ns)}")
    print()

    # Quick ratios (higher is worse)
    print(f"FastTimer / baseline: {fast_ns / baseline:.2f}x")
    print(f"TimerCtx / baseline: {timer_ns / baseline:.2f}x")
    if fast_ns > 0:
        print(f"TimerCtx / FastTimer: {timer_ns / fast_ns:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
