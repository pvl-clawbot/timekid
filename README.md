# Timekid

A high-precision timing and profiling library for Python with support for context managers, decorators, lap timing, and benchmarking.

[![Tests](https://github.com/pvl-clawbot/timekid/actions/workflows/test.yml/badge.svg)](https://github.com/pvl-clawbot/timekid/actions/workflows/test.yml)

## Features

- **Multiple Timer Interfaces**: Choose between `StopWatch`, `TimerContext`, or the registry-based `Timer` class
- **Context Manager Support**: Time code blocks with clean `with` statements
- **Decorator Support**: Automatically time function executions with `@timer.timed` and `@timer.timed_async`
- **Lap Timing**: Record intermediate times within a timing context
- **Historical Tracking**: All timing invocations are stored in lists for statistical analysis
- **Benchmarking**: Run functions multiple times with warmup support
- **Async Support**: Full support for async functions with `@timer.timed_async`
- **Type Safe**: Comprehensive type hints using modern Python typing features
- **Zero Dependencies**: Uses only Python standard library
- **High Precision**: Uses `time.perf_counter()` for accurate timing

## Installation

**Requirements:** Python 3.12+

Install with uv (recommended):
```bash
uv pip install -e .
python -c "import timekid; print('timekid import OK')"
```

Or with pip:
```bash
pip install -e .
python -c "import timekid; print('timekid import OK')"
```

Verify the expected import works:
```bash
python -c "import timekid; print('timekid OK')"
```

## Quick Start

### Low-overhead timing (FastTimer)

If you need minimal overhead and are doing a large number of measurements, use `FastTimer`.
It stores raw integer nanoseconds internally and only converts to seconds when you report.

```python
from timekid.fast import FastTimer

ft = FastTimer()
key = ft.key_id("hot_loop")  # do string->id once

for _ in range(1000):
    tok = ft.start(key)
    # ... hot code ...
    ft.stop(tok)

print(ft.times_s(precision=6)[key][:5])
```

(Planned) a future optional Rust backend can implement the same API for even lower overhead.

### Basic Timing with Context Manager

```python
from timekid.timer import Timer

timer = Timer(precision=3)

with timer['database_query']:
    # Your code here
    result = execute_query()

# Access timing (returns list of floats)
print(f"Query took {timer.times['database_query'][0]}s")
```

### Function Timing with Decorators

```python
from timekid.timer import Timer

timer = Timer(precision=2)

@timer.timed
def process_data(data):
    # Your processing logic
    return processed_data

# Call multiple times
for item in items:
    process_data(item)

# Analyze all invocations
times = timer.times['process_data']
print(f"Average: {sum(times) / len(times):.3f}s")
print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")
```

### Lap Timing

```python
with timer['data_pipeline'] as t:
    load_data()
    t.lap()  # Record lap 1

    transform_data()
    t.lap()  # Record lap 2

    save_data()
    # Final lap recorded automatically on exit

# Access lap times
contexts = timer.get('data_pipeline')
print(f"Load: {contexts[0].laps[0]}s")
print(f"Transform: {contexts[0].laps[1]}s")
print(f"Save: {contexts[0].laps[2]}s")
```

### Async Function Timing

```python
import asyncio
from timekid.timer import Timer

timer = Timer()

@timer.timed_async
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Use normally
await fetch_data('https://api.example.com/data')
```

### Benchmarking

```python
from timekid.timer import Timer

timer = Timer()

# Benchmark a function with 1000 iterations
results = timer.benchmark(my_function, num_iter=1000, arg1, arg2)

# Optionally persist benchmark runs in the timer registry
timer.benchmark(my_function, num_iter=1000, arg1, arg2, store=True)
print(len(timer.times['my_function benchmark']))

# Optionally provide a custom registry key when storing
custom_key = 'bench.my_function.hot_path'
timer.benchmark(my_function, num_iter=1000, arg1, arg2, store=True, key=custom_key)
print(len(timer.times[custom_key]))

# Analyze results
times = [r.elapsed_time for r in results]
avg_time = sum(times) / len(times)
print(f"Average: {avg_time:.6f}s")
```

### Simple StopWatch

```python
from timekid.timer import StopWatch

sw = StopWatch(precision=2)
sw.start()

# Your code here
do_something()

elapsed = sw.stop()
print(f"Elapsed: {elapsed}s")
```

## Key Concepts

### List-Based Registry

All timer registry values are stored as **lists of TimerContext objects**. This enables:

- **Historical tracking**: Every invocation is preserved
- **Statistical analysis**: Calculate min, max, average, standard deviation
- **Performance trends**: Track performance over time
- **Consistent API**: No mix of single values and lists

```python
# Multiple invocations create list entries
with timer['task']:
    do_work()

with timer['task']:
    do_work()

# Access all timings
all_times = timer.times['task']  # Returns list[float]
first_time = timer.times['task'][0]
latest_time = timer.times['task'][-1]
```

### Decorator Behavior

Decorated functions store all invocations under the function name (no numbered keys):

```python
@timer.timed
def process(item):
    return item * 2

# Call 3 times
process(1)
process(2)
process(3)

# All stored under 'process' key
print(len(timer.times['process']))  # Output: 3
```

### Precision Control

All timer types accept an optional `precision` parameter for rounding:

```python
timer = Timer(precision=3)  # Round to 3 decimal places

with timer['task']:
    time.sleep(0.123456)

print(timer.times['task'][0])  # Output: 0.123
```

### Verbose Mode

Enable verbose logging to see timing events in real-time:

```python
import logging

logger = logging.getLogger(__name__)
timer = Timer(verbose=True, log_func=logger.info)

with timer['task']:
    # Logs start and stop events
    do_work()
```

## API Reference

### Timer Class

**Main registry-based interface for timing operations.**

```python
Timer(precision: Optional[int] = None, verbose: bool = False, log_func: Callable[[str], None] = print)
```

**Properties:**
- `times: Dict[str, list[float]]` - All elapsed times for succeeded timers
- `contexts: Dict[str, list[TimerContext]]` - All timer contexts
- `precision: Optional[int]` - Configured precision for rounding

**Methods:**
- `timer['key']` - Create/access timer context (creates new context each time)
- `timed(func)` - Decorator for synchronous functions
- `timed_async(func)` - Decorator for async functions
- `get(key: str)` - Get all contexts matching a key
- `status(key: str)` - Get list of statuses for a key
- `sorted(reverse: bool = False)` - Get timers sorted by elapsed time
- `timeit(func, *args, **kwargs)` - Time a single function call (method name; unrelated to Python's `timeit` module)
- `benchmark(func, num_iter: int, *args, store: bool = False, key: Optional[str] = None, **kwargs)` - Benchmark function with multiple iterations (set `store=True` to persist results; set `key` to control the registry key)
- `anonymous(name, verbose, log_func)` - Create anonymous timer context (not stored in registry)

> Note: `Timer.timeit(...)` is an instance method on `Timer`; it does not wrap or proxy the stdlib `timeit` module.

### TimerContext Class

**Context manager for timing code blocks.**

```python
TimerContext(precision: Optional[int], name: Optional[str] = None, verbose: bool = False, log_func: Callable[[str], None] = print)
```

**Properties:**
- `elapsed_time: float` - Total elapsed time
- `laps: list[float]` - List of lap times
- `status: Status` - Current status (PENDING/RUNNING/SUCCEEDED/FAILED)
- `name: str` - Timer name

**Methods:**
- `lap()` - Record intermediate time
- `reset()` - Reset timer (clears laps, starts from now)
- `rename(name: str)` - Change timer name

### StopWatch Class

**Simple imperative timer with manual control.**

```python
StopWatch(precision: Optional[int] = None)
```

**Properties:**
- `elapsed_time: float` - Elapsed time (raises error if not started)
- `status: Status` - Current status

**Methods:**
- `start()` - Start timing
- `stop()` - Stop timing and return elapsed time
- `reset()` - Reset to initial state

### Status Enum

Timer lifecycle states:
- `Status.PENDING` - Created but not started
- `Status.RUNNING` - Currently timing
- `Status.STOPPED` - Manually stopped (StopWatch only)
- `Status.SUCCEEDED` - Context exited normally
- `Status.FAILED` - Context exited with exception

## Testing

Run tests with unittest:
```bash
python -m unittest tests._basic_test -v
```

## Development

This project uses `uv` for package management:

```bash
# Install in editable mode
uv pip install -e .

# Run tests
.venv/bin/python -m unittest tests._basic_test -v

# Run examples
python -m timekid.timer
```

## CI/CD

The project includes GitHub Actions workflow for automated testing:
- Runs on Python 3.12 and 3.13
- Tests on push/PR to main, master, or develop branches
- Uses `uv` for dependency management

## Migration from Older Versions

If upgrading from a version with mixed single/list registry values:

```python
# Old way (single values)
elapsed = timer.times['my_task']  # Was a float

# New way (list values)
elapsed = timer.times['my_task'][0]  # First timing
elapsed = timer.times['my_task'][-1]  # Latest timing
```

## Contributing

Contributions are welcome! Please ensure:
- Python 3.12+ compatibility
- All tests pass
- Type hints for all functions
- Update documentation as needed

## License

MIT License

## Author

Peter Vestereng Larsen (p.vesterenglarsen@gmail.com)
