# Issue draft: Sub-microsecond / lower-overhead measurement mode

## Motivation
Timekid aims to be a "gold standard" timing/performance evaluation library.
For very small code paths (microsecond and below), Python overhead (context manager, object creation, dict lookup, list append, wrapper calls) can dominate the measurement and distort results.

## Proposal
Add a low-overhead API surface (and eventual native backend) optimized for high-volume measurements:

- A `FastTimer` that stores raw integer nanoseconds internally.
- Optional `key_id("name") -> int` to do string hashing/dict work once.
- `start(key) -> token` / `stop(token) -> elapsed_ns` hot path.
- Convert to seconds + rounding only at reporting time.

## Notes on precision
- Python already exposes `time.perf_counter_ns()` (ns units, not necessarily ns *accuracy*).
- On modern Linux/macOS/Windows, accuracy is usually in the 10s–100s of ns to a few microseconds depending on platform.
- For honest runtimes, the key is lowering *observer overhead* more than chasing nominal ns.

## Future (optional)
- Rust backend via PyO3 + maturin:
  - store deltas in `Vec<u64>` per key
  - bulk-export results to Python
  - optional ring buffer mode

## Acceptance criteria
- Demonstrate reduced overhead in a microbenchmark vs the current `Timer` context-manager path.
- Maintain pure-Python fallback.
