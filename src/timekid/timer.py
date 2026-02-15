import time
import logging
from types import TracebackType
from typing import (Self, Type, Optional, Callable, Generator,
                    Awaitable, ParamSpec, TypeVar)
from functools import wraps
from enum import StrEnum
from contextlib import contextmanager

__author__ = "Peter Vestereng Larsen"
__version__ = "0.1.0"
__license__ = "MIT"
__email__ = "p.vesterenglarsen@gmail.com"

__all__ = ['Timer', 'StopWatch', 'TimerContext', 'Status', 'BaseTimer']
logger = logging.getLogger('timekid')

P = ParamSpec(name='P')
P_async = ParamSpec(name='P_async')

R = TypeVar(name='R')
R_async = TypeVar(name='R_async')


def _noop(_: str) -> None:
    """No-op function for verbose=False mode."""
    return None


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
        if self._elapsed_time is not None:
            return self._elapsed_time
        if self._start_time is None:
            raise AttributeError("Elapsed time is not available until the timer has been started.")
        # Return rounded value but don't cache during running state
        return self._round(time.perf_counter() - self._start_time)

    @property
    def status(self) -> Status:
        return self._status
        
    def start(self) -> None:
        self._end_time = None
        self._elapsed_time = None
        self._status = Status.RUNNING
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Raises:
            AttributeError: If timer hasn't been started
            RuntimeError: If timer is already stopped
        """
        if self._start_time is None:
            raise AttributeError("Cannot stop timer before it has started.")
        if self._status == Status.STOPPED:
            raise RuntimeError("Timer is already stopped.")

        self._end_time = time.perf_counter()
        self._elapsed_time = self._round(self._end_time - self._start_time)
        self._status = Status.STOPPED
        return self._elapsed_time
    
    def reset(self) -> None:
        self._start_time = None
        self._end_time = None
        self._elapsed_time = None
        self._status = Status.PENDING

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.stop()
        if exc_type is not None:
            self._status = Status.FAILED

    def __repr__(self) -> str:
        parts = [f'status={self.status}']
        if self._start_time is not None:
            parts.insert(0, f'elapsed_time={self.elapsed_time}s')
        return f"StopWatch({', '.join(parts)})"

class TimerContext(BaseTimer):
    """TimerContext is a context manager for timing code execution.
    It tracks elapsed time, lap times, and status of the timer.
    This object should only be used as a context manager.
    It is not intended to be used outside of a with statement.
    While TimerContext can be used directly, the recommended approach is to access it via the Timer class (e.g., `timer["my_key"]`).
    """
    def __init__(self, precision: Optional[int], name: Optional[str] = None, verbose: bool = False,
                 log_func: Callable[[str], None] = logger.info) -> None:
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
        self._log_func: Callable[[str], None] = log_func if verbose else _noop
    
    @property
    def elapsed_time(self) -> float:
        if self._elapsed_time is not None:
            return self._elapsed_time
        if self._start_time is None:
            raise AttributeError("Elapsed time is not available until the timer has been started.")
        # Return rounded value but don't cache during running state
        return self._round(time.perf_counter() - self._start_time)

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
        """Reset the timer to start timing from now, clearing all previous lap data."""
        self._start_time = time.perf_counter()
        self._lap_start_time = self._start_time
        self._elapsed_time = None
        self._laps.clear()
        self._has_laps = False
    
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
        self._status = Status.RUNNING
        self._entered = True
        self._start_time = time.perf_counter()
        self._lap_start_time = self._start_time
        return self
    
    def __exit__(self,
                    exc_type: Optional[Type[BaseException]], 
                    exc_value: Optional[BaseException], 
                    traceback: Optional[TracebackType]) -> None:
        end_time = time.perf_counter()
        if self._start_time is None or self._lap_start_time is None:
            raise AttributeError("Cannot exit context before a time has been recorded.")
        self._elapsed_time = self._round(end_time - self._start_time)
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
            message = f"Timer '{self._name}' finished with status: {self._status}"
            if exc_value:
                message += f", exception: {exc_value}"
            self._log_func(message)
            if self._precision is not None:
                self._log_func(f"Elapsed time: {self.elapsed_time:.{self._precision}f}s")
            else:
                self._log_func(f"Elapsed time: {self.elapsed_time}s")
            self._log_func(f"Status: {self._status}")
    
    def __repr__(self) -> str:
        parts = [f'name={self._name}']
        if self._entered:
            parts.append(f'elapsed_time={self.elapsed_time}s')
            if self._has_laps:
                parts.append(f'laps={self.laps}')
        parts.append(f'status={self._status}')
        return f"TimerContext({', '.join(parts)})"

class Timer:
    def __init__(self, precision: Optional[int] = None, verbose: bool = False,
                 log_func: Callable[[str], None] = logger.info) -> None:
        self._precision = precision
        self._verbose = verbose
        self._log_func = log_func
        self._registry: dict[str, list[TimerContext]] = {}

    @property
    def precision(self) -> Optional[int]:
        """The precision (decimal places) for rounding timing values."""
        return self._precision

    @property
    def times(self) -> dict[str, list[float]]:
        """Return dict of elapsed times for all succeeded/stopped timers.

        Always returns lists of floats, even if only one timing exists for a key.
        Only includes timers with SUCCEEDED or STOPPED status.

        Returns:
            Dict mapping timer keys to lists of elapsed times
        """
        result: dict[str, list[float]] = {}
        for key, contexts in self._registry.items():
            # Get all times for succeeded/stopped timers
            times_list = [ctx.elapsed_time for ctx in contexts
                         if ctx.status in (Status.SUCCEEDED, Status.STOPPED)]
            if times_list:
                result[key] = times_list
        return result
    
    @property
    def contexts(self) -> dict[str, list[TimerContext]]:
        """Return dict of all timer contexts.

        Always returns lists of TimerContext objects, even if only one exists for a key.

        Returns:
            Dict mapping timer keys to lists of TimerContext objects
        """
        return dict(self._registry)
    
    def status(self, key: str) -> list[Status]:
        """Get the status of all timer contexts for a given key.

        Always returns a list of Status values, even if only one context exists.

        Args:
            key: The registry key to look up

        Returns:
            List of Status values for all contexts under this key

        Raises:
            KeyError: If key is not found in registry
        """
        if key not in self._registry:
            raise KeyError(f"Key '{key}' not found in timer registry.")
        return [ctx.status for ctx in self._registry[key]]
    
    def timed(self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorator to time function execution.

        Each invocation creates a new TimerContext and appends it to the list
        stored under the function's name in the registry.

        Args:
            func: The function to time

        Returns:
            Wrapped function that times execution
        """
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key: str = func.__name__
            ctx = self[key]
            with ctx:
                result: R = func(*args, **kwargs)
            return result
        return wrapper
    
    def timed_async(self, func: Callable[P_async, Awaitable[R_async]]) -> Callable[P_async, Awaitable[R_async]]:
        """Decorator to time async function execution.

        Each invocation creates a new TimerContext and appends it to the list
        stored under the function's name in the registry.

        Args:
            func: The async function to time

        Returns:
            Wrapped async function that times execution
        """
        @wraps(func)
        async def wrapper(*args: P_async.args, **kwargs: P_async.kwargs) -> R_async:
            key: str = func.__name__
            ctx = self[key]
            with ctx:
                result: R_async = await func(*args, **kwargs)
            return result

        return wrapper
    
    def get(self, key: str) -> list[TimerContext]:
        """Get all timer contexts matching a key.

        Returns all TimerContext objects from the given key.
        If the key does not exist, returns an empty list.

        Args:
            key: The registry key

        Returns:
            List of all matching TimerContext objects
        """
        return self._registry.get(key, [])
    
    @contextmanager
    def anonymous(self, name: Optional[str] = None, verbose: bool = False, log_func: Callable[[str], None] = logger.info) -> Generator[TimerContext, None, None]:
        with TimerContext(self.precision, name=name, verbose=verbose, log_func=log_func) as ctx:
            yield ctx
            
    def sorted(self, reverse: bool = False) -> list[tuple[str, TimerContext]]:
        """Get all succeeded timers sorted by elapsed time.

        Uses the most recent (last) context for each key when sorting.
        Only includes contexts with SUCCEEDED status.

        Args:
            reverse: If True, sort from slowest to fastest. Default is fastest to slowest.

        Returns:
            List of (key, context) tuples sorted by elapsed time
        """
        result: list[tuple[str, TimerContext]] = []
        for key, contexts in self._registry.items():
            # Get the most recent succeeded context
            succeeded_contexts = [ctx for ctx in contexts if ctx.status == Status.SUCCEEDED]
            if succeeded_contexts:
                # Use the most recent one (last in list)
                result.append((key, succeeded_contexts[-1]))

        return sorted(result, key=lambda item: item[1].elapsed_time, reverse=reverse)
        
    def timeit(self, func: Callable[P, R],
               *args: P.args, **kwargs: P.kwargs) -> TimerContext:
        """Time a single invocation of a function.

        Similar to the timed decorator but for one-off timing.
        Creates a timer with key based on function name.

        Args:
            func: The function to time
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            The TimerContext containing timing information
        """
        key: str = f"{func.__name__}"
        with self[key] as ctx:
            func(*args, **kwargs)
        return ctx
    
    def benchmark(self, func: Callable[P, R], num_iter: int, warmup: int = 1,
                  *args: P.args, **kwargs: P.kwargs) -> list[TimerContext]:
        # Benchmark runs anonymously and doesn't persist in registry
        # Warmup runs to handle JIT compilation or lazy initialization
        for _ in range(warmup):
            func(*args, **kwargs)

        str_key: str = f"{func.__name__} benchmark"
        results: list[TimerContext] = []
        for _ in range(num_iter):
            with self.anonymous(name = str_key) as ctx:
                func(*args, **kwargs)
                results.append(ctx)
        return results
        
    def __getitem__(self, key: str) -> TimerContext:
        """Get or create a TimerContext for the given key.

        If the key doesn't exist, creates a new list with one TimerContext.
        If the key exists, creates a new TimerContext and appends it to the list,
        then returns this new context.

        This allows `with timer['key']:` to work naturally while storing all
        invocations in a list.

        Args:
            key: The registry key

        Returns:
            A new TimerContext instance for this invocation
        """
        # Create a new context for this usage
        context = TimerContext(precision=self._precision, name=key,
                              verbose=self._verbose, log_func=self._log_func)

        # Initialize list if this is the first time
        if key not in self._registry:
            self._registry[key] = []

        # Append the new context to the list
        self._registry[key].append(context)

        return context

    # (removed) _async_context: unused internal helper; async timings use sync TimerContext.

    def __repr__(self) -> str:
        return f"Timer(precision={self.precision}, times={self.times})"