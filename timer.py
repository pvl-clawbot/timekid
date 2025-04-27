import time
from types import TracebackType
from typing import (Self, Any, Type, Optional, Callable, Generator, 
                    Hashable, Awaitable, AsyncGenerator, ParamSpec, TypeVar)
from collections import defaultdict
from functools import wraps
from enum import StrEnum
from contextlib import contextmanager, asynccontextmanager
import logging
from itertools import repeat

__author__ = "Peter Vestereng Larsen"
__version__ = "0.1.0"
__license__ = "MIT" #depending on git
__email__ = "p.vesterenglarsen@gmail.com"

P = ParamSpec(name='P')
P_async = ParamSpec(name='P_async')

R = TypeVar(name='R')
R_async = TypeVar(name='R_async')

Q = ParamSpec(name='Q')
S = TypeVar(name='S')

T = ParamSpec(name='T')
V = TypeVar(name='V')


class Status(StrEnum):
    """Enum object indicating the status of a timer."""
    PENDING = 'pending'
    RUNNING = 'running'
    STOPPED = 'stopped'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    
class BaseTimer:
    """Base class for timers. This class is not intended to be used directly."""
    _precision: Optional[int]
    
    def _round(self, value: float) -> float:
        """Round the value to the specified precision. If precision is None, return the value as is.

        Args:
            value (float): the value to round

        Returns:
            float: the rounded value
        """
        if self._precision is None:
            return value
        return round(value, self._precision)

class StopWatch(BaseTimer):
    def __init__(self, precision: Optional[int] = None) -> None:
        self._elapsed_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._precision = precision
        self._status: Status = Status.PENDING
        
    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            raise AttributeError("Elapsed time is not available until the timer has been started.")
        if self._elapsed_time is None:
            return self._round(time.perf_counter() - self._start_time)
        return self._round(self._elapsed_time)
    
    @property
    def duration(self) -> float:
        return self.elapsed_time
    
    @property
    def status(self) -> Status:
        return self._status
        
    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._end_time = None
        self._elapsed_time = None
        self._status = Status.RUNNING
    
    def stop(self) -> float:
        self._end_time = time.perf_counter()
        if self._start_time is None:
            raise AttributeError("Cannot stop timer before it has started.")
        self._elapsed_time = self._round(self._end_time - self._start_time)
        self._status = Status.STOPPED
        return self._elapsed_time
    
    def reset(self) -> None:
        self._start_time = None
        self._end_time = None
        self._elapsed_time = None
        self._status = Status.PENDING
        
    def __repr__(self) -> str:
        if self._start_time is None:
            return f'StopWatch(status={self.status})'
        return f'StopWatch(elapsed_time={self.elapsed_time}s, status={self.status})'

class TimerContext(BaseTimer):
    """TimerContext is a context manager for timing code execution.
    It tracks elapsed time, lap times, and status of the timer.
    This object should only be used as a context manager.
    It is not intended to be used outside of a with statement.
    While TimerContext can be used directly, the recommended approach is to access it via the Timer class (e.g., `timer["my_key"]`).
    """
    def __init__(self, precision: Optional[int], name: Optional[str] = None, verbose: bool = False,
                 log_func: Callable[[str],Any] = print) -> None:
        self._name: str = str(name) if name is not None else 'Unnamed'
        self._precision = precision
        self._elapsed_time: Optional[float] = None
        self._laps: list[float] = []
        self._lap_start_time: Optional[float] = None
        self._has_laps: bool = False
        self._start_time: Optional[float] = None
        self._status: Status = Status.PENDING
        self._entered: bool = False
        self._verbose: bool = verbose
        self._log_func = log_func if verbose else (lambda _: None)
    
    @property
    def elapsed_time(self) -> float:
        if self._elapsed_time is None:
            if self._start_time is not None:
                return self._round(time.perf_counter() - self._start_time)
            raise AttributeError("Elapsed time is not available until the timer has been started.")
        return self._round(self._elapsed_time)
    
    @property
    def duration(self) -> float:
        return self.elapsed_time
    
    @elapsed_time.setter
    def elapsed_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("Elapsed time cannot be negative.")
        self._elapsed_time = self._round(value)
        
    @property
    def laps(self) -> list[float]:
        return self._laps
    
    @property
    def status(self) -> Status:
        return self._status
    
    @property
    def name(self) -> str:
        return self._name
    
    def reset(self) -> None:
        self._start_time = time.perf_counter()
        self._lap_start_time = self._start_time
    
    def rename(self, name: str) -> None:
        self._name = name
    
    def lap(self) -> None:
        if self._lap_start_time is None:
            raise AttributeError("Cannot lap before a time has been recorded.")
        current_time = time.perf_counter()
        self._laps.append(self._round(current_time-self._lap_start_time))
        self._lap_start_time = current_time
        self._has_laps = True
        if self._verbose:
            self._log_func(f"Lap {len(self._laps)} time: {self._laps[-1]}s")
    
    def __enter__(self) -> Self:
        if self._entered and self._verbose:
            self._log_func(f'Timer {self._name} is being overwritten.')
        self._start_time = time.perf_counter()
        self._lap_start_time = self._start_time
        self._status = Status.RUNNING
        self._entered = True
        return self
    
    def __exit__(self,
                    exc_type: Optional[Type[BaseException]], 
                    exc_value: Optional[BaseException], 
                    traceback: Optional[TracebackType]) -> None:
        end_time = time.perf_counter()
        if self._start_time is None or self._lap_start_time is None:
            raise AttributeError("Cannot exit context before a time has been recorded.")
        self._elapsed_time = end_time - self._start_time
        last_lap_duration: float = end_time - self._lap_start_time
        self._laps.append(self._round(last_lap_duration))
        self._lap_start_time = None
        if exc_type is None:
            self._status = Status.SUCCEEDED
        else:
            self._status = Status.FAILED
        if self._verbose:
            if self._has_laps:
                self._log_func(f"Lap {len(self._laps)} time: {self._laps[-1]}s")
            self._log_func(f"Timer '{self._name}' finished with status: {self._status}" + f", exception: {exc_value}" if exc_value else "")
            if self._precision is not None:
                self._log_func(f"Elapsed time: {self.elapsed_time:.{self._precision}f}s")
            else:
                self._log_func(f"Elapsed time: {self.elapsed_time}s")
            self._log_func(f"Status: {self._status}")
    
    def __repr__(self) -> str:
        if not self._entered:
            return f"TimerContext(name={self._name}, status={self._status})"
        if not self._has_laps:
            return f"TimerContext(name={self._name}, elapsed_time={self.elapsed_time}s, status={self._status})"
        return f"TimerContext(name={self._name}, elapsed_time={self.elapsed_time}s, laps={self.laps}, status={self._status})"

class Timer:
    def __init__(self, precision: Optional[int] = None, verbose: bool = False,
                 log_func: Callable[[str], Any] = print) -> None:
        self.precision = precision
        self._updated: bool = False
        self._times: dict[str, float] = {}
        self._keys_to_update: set[str] = set()
        self._call_counters: dict[str, int] = defaultdict(int)
        self._registry: dict[str, TimerContext] = defaultdict(lambda: TimerContext(precision=precision, verbose=verbose, 
                                                                                   log_func = log_func))
        
    @property
    def times(self) -> dict[str, float]:
        if not self._updated:
            self._update_times()
        return self._times
    
    @property
    def contexts(self) -> dict[str, TimerContext]:
        return dict(self._registry)
    
    def status(self, key: Hashable) -> Status:
        str_key: str = str(key)
        if str_key not in self._registry:
            raise KeyError(f"Key '{str_key}' not found in timer registry.")
        return self._registry[str_key].status
    
    def timed(self, func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            count = self._call_counters[func.__name__]
            key: str = f'{func.__name__} ({count})'
            self._call_counters[func.__name__] += 1
            with self[key]:
                result: R = func(*args, **kwargs)
            return result
        return wrapper
    
    def timed_async(self, func: Callable[P_async, Awaitable[R_async]]) -> Callable[P_async, Awaitable[R_async]]:
        @wraps(func)
        async def wrapper(*args: P_async.args, **kwargs: P_async.kwargs) -> R_async:
            count = self._call_counters[func.__name__]
            async_key: str = f'{func.__name__} ({count})'
            self._call_counters[func.__name__] += 1
            async with self._async_context(async_key):
                return await func(*args, **kwargs)

        return wrapper
    
    def get(self, partial_key: Hashable) -> list[TimerContext]:
        str_partial: str = str(partial_key)
        result: list[TimerContext] = []
        for key, context in self._registry.items():
            if str_partial in key:
                result.append(context)
        return result
    
    @contextmanager
    def anonymous(self, name: Optional[Hashable] = None, verbose: bool = False, log_func: Callable[[str], Any] = print) -> Generator[TimerContext, None, None]:
        str_name: str = str(name)
        with TimerContext(self.precision, name=str_name, verbose=verbose, log_func=log_func) as ctx:
            yield ctx
            
    def sorted(self, reverse: bool = False) -> list[tuple[str, TimerContext]]:
        return sorted(
            [(k, v) for k, v in self._registry.items() if v.status == Status.SUCCEEDED],
            key=lambda item: item[1].elapsed_time,
            reverse=reverse
        )
        
    def timeit(self, func: Callable[Q, S],
               *args: Q.args, **kwargs: Q.kwargs) -> TimerContext:
        count = self._call_counters[f"{func.__name__} timeit"]
        key: str = f"{func.__name__} timeit ({count})"
        self._call_counters[f"{func.__name__} timeit"] += 1
        with self[key] as ctx:
            func(*args, **kwargs)
        return ctx
    
    def benchmark(self, func: Callable[T, V], num_iter: int,
                  *args: T.args, **kwargs: T.kwargs) -> list[TimerContext]:
        "[DEV NOTE] Benchmark runs anonymously."
        str_key: str = f"{func.__name__} benchmark"
        results: list[TimerContext] = []
        for _ in repeat(None, num_iter):
            with self.anonymous(name = str_key) as ctx:
                func(*args, **kwargs)
                results.append(ctx)
        return results
        
    def __getitem__(self, key: Hashable) -> TimerContext:
        str_key: str = str(key)
        if str_key not in self._registry:
            self._updated = False
        context: TimerContext  = self._registry[str_key]
        if not self._updated:
            context.rename(str_key)
            self._keys_to_update.add(str_key)
        return context
    
    def _update_times(self) -> None:
        for key in self._keys_to_update:
            context: TimerContext = self._registry[key]
            if context._elapsed_time is not None:
                self._times[key] = context.elapsed_time
        self._updated = True
        self._keys_to_update.clear()

    @asynccontextmanager
    async def _async_context(self, key: str) -> AsyncGenerator[TimerContext, None]:
        str_key: str = str(key)
        ctx = self[str_key]
        ctx.__enter__()
        try:
            yield ctx
        except Exception as e:
            ctx.__exit__(type(e), e, e.__traceback__)
            raise
        else:
            ctx.__exit__(None, None, None)
        
    def __repr__(self) -> str:
        return f"Timer(precision={self.precision}, times={self.times})"
    
    
if __name__ == '__main__':
    timer = Timer(precision=2)
    with timer['test'] as t:
        time.sleep(.15)
    print(timer['test'])
    with timer['test2'] as t:
        time.sleep(.29)
    print(timer['test2'])
    with timer['test3'] as t:
        time.sleep(.369)
    print(timer['test3'])
    with timer['loop'] as t:
        for _ in range(1_000_000):
            pass
    print(timer['loop'])
    print(timer.times)
    
    with timer['reset test'] as t:
        time.sleep(1)
        t.reset()
        time.sleep(2)
    
    print(timer['reset test'], 'should be around 2 seconds')
    
    timer = Timer(verbose=True)
    with timer['lap test'] as t:
        time.sleep(1)
        t.lap()
        time.sleep(2)
        t.lap()
        time.sleep(3)
    
    print(timer['lap test'])
    print(timer['lap test'].laps)
    print(sum(timer['lap test'].laps), "should be equal to elapsed time:", timer['lap test'].elapsed_time)
    
    timer = Timer(precision=2)
    with timer['loop test'] as t:
        for _ in range(99):
            time.sleep(0.01)
            t.lap()
        time.sleep(0.1)
        print(t)
    print(timer['loop test'])
    print(timer)
    
    
    
    @timer.timed
    def foo(x: float) -> float:
        time.sleep(x)
        return x
    
    @timer.timed
    def bar(x: float) -> float:
        time.sleep(x)
        return x
    
    
    foo(0.1)
    bar(0.15)
    foo(0.2)
    bar(0.25)
    foo(0.3)
    bar(0.35)
    print(timer.times)
    print(timer.times)
    
    @timer.timed
    def baz(x: float) -> float:
        time.sleep(x)
        return x
    baz(0.1)
    print(timer.times)
    print(timer.times)
    
    with timer.anonymous(verbose=True) as ctx:
        time.sleep(0.1)
        print(ctx)
    print(ctx)
    
    with timer.anonymous(name='NamedAnon') as ctx:
        time.sleep(0.1)
        print(ctx)
    print(ctx)
    
    with timer['elapsed_time'] as t:
        time.sleep(0.1)
        print(t.elapsed_time, t)
        print(timer.status('elapsed_time'), t.status)
        time.sleep(0.1)
    print(t.elapsed_time, t)
    time.sleep(0.1)
    print(t.elapsed_time, t)
    
    print(timer['uninstantiated'])
    print('something else')
    print(timer.get('foo'))
    
    print(timer.sorted())
    
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    timer = Timer(verbose=True, log_func=logger.info)
    with timer['log_test'] as t:
        time.sleep(0.1)
        t.lap()
        time.sleep(0.1)
    
    timer = Timer(verbose=True, log_func=logger.critical, precision=2)
    with timer['log_test'] as t:
        time.sleep(0.1)
        t.lap()
        time.sleep(0.1)
    
    @timer.timed
    def log_foo(x: float) -> None:
        time.sleep(x)
        
    log_foo(0.1)
    log_foo(0.2)
    
    print(timer.get('log_foo'))
    
    print(timer.contexts)
    
    sw = StopWatch()
    print(sw)
    sw.start()
    time.sleep(0.1)
    print(sw)
    print(sw.stop())
    print(sw)
    
    @timer.timed
    def foo_error(x: float) -> float:
        time.sleep(x / 2)
        raise ValueError("This is an error")
        time.sleep(x / 2)
    try:
        foo_error(0.1)
    except Exception as e:
        print(f"Exception caught: {e}")
        pass
    print(timer.get('foo_error'))
    
    print(timer.timeit(foo, 0.1))
    print(timer.timeit(foo, x=0.1))
    
    bench_results = timer.benchmark(foo, 10, 0.1)
    print(bench_results)