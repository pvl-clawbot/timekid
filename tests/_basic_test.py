import time
import unittest
import asyncio
import warnings

from timekid.timer import TimerContext, Timer, Status, StopWatch

class TestTimerContext(unittest.TestCase):
    def test_elapsed_time(self):
        with TimerContext(precision=2) as timer:
            time.sleep(0.1)
        self.assertAlmostEqual(timer.elapsed_time, 0.1, places=1)

    def test_reset(self):
        with TimerContext(precision=2) as timer:
            time.sleep(0.1)
            timer.reset()
            time.sleep(0.2)
        self.assertAlmostEqual(timer.elapsed_time, 0.2, places=1)

    def test_lap(self):
        with TimerContext(precision=2) as timer:
            time.sleep(0.1)
            timer.lap()
            time.sleep(0.2)
            timer.lap()
            time.sleep(0.3)
        self.assertAlmostEqual(sum(timer.laps), timer.elapsed_time, places=1)
        self.assertAlmostEqual(timer.laps[0], 0.1, places=1)
        self.assertAlmostEqual(timer.laps[1], 0.2, places=1)

    def test_elapsed_time_before_recording(self):
        timer = TimerContext(precision=2)
        with self.assertRaises(AttributeError):
            _ = timer.elapsed_time

    def test_negative_elapsed_time(self):
        timer = TimerContext(precision=2)
        with self.assertRaises(AttributeError):
            timer.elapsed_time = -1 # type: ignore (ignore type for testing purposes)

    def test_repr(self):
        with TimerContext(precision=2) as timer:
            time.sleep(0.1)
        self.assertIn("TimerContext(name=Unnamed, elapsed_time=", repr(timer))

    def test_precision_none(self):
        with TimerContext(precision=None) as timer:
            time.sleep(0.123456)
        self.assertAlmostEqual(timer.elapsed_time, 0.123456, places=2)


class TestStopWatch(unittest.TestCase):
    def test_equality_pending(self):
        sw1 = StopWatch(precision=2)
        sw2 = StopWatch(precision=2)
        self.assertEqual(sw1, sw2)

        sw3 = StopWatch(precision=3)
        self.assertNotEqual(sw1, sw3)

    def test_basic_timing(self):
        """Test basic start/stop functionality."""
        sw = StopWatch(precision=2)
        self.assertEqual(sw.status, Status.PENDING)

        sw.start()
        self.assertEqual(sw.status, Status.RUNNING)
        time.sleep(0.1)
        elapsed = sw.stop()

        self.assertEqual(sw.status, Status.STOPPED)
        self.assertAlmostEqual(elapsed, 0.1, places=1)
        self.assertAlmostEqual(sw.elapsed_time, 0.1, places=1)

    def test_stop_twice_raises_error(self):
        """Test that calling stop() twice raises RuntimeError."""
        sw = StopWatch()
        sw.start()
        time.sleep(0.05)
        sw.stop()

        with self.assertRaises(RuntimeError) as context:
            sw.stop()
        self.assertIn("already stopped", str(context.exception))

    def test_elapsed_time_before_start_raises_error(self):
        """Test that accessing elapsed_time before start() raises AttributeError."""
        sw = StopWatch()
        with self.assertRaises(AttributeError) as context:
            _ = sw.elapsed_time
        self.assertIn("not available until", str(context.exception))

    def test_stop_before_start_raises_error(self):
        """Test that calling stop() before start() raises AttributeError."""
        sw = StopWatch()
        with self.assertRaises(AttributeError) as context:
            sw.stop()
        self.assertIn("before it has started", str(context.exception))

    def test_reset_functionality(self):
        """Test that reset() properly clears timer state."""
        sw = StopWatch(precision=2)

        # First timing
        sw.start()
        time.sleep(0.1)
        sw.stop()
        first_time = sw.elapsed_time

        # Reset and time again
        sw.reset()
        self.assertEqual(sw.status, Status.PENDING)

        sw.start()
        time.sleep(0.2)
        sw.stop()
        second_time = sw.elapsed_time

        self.assertAlmostEqual(first_time, 0.1, places=1)
        self.assertAlmostEqual(second_time, 0.2, places=1)
        self.assertNotEqual(first_time, second_time)

    def test_elapsed_time_accuracy(self):
        """Test that elapsed_time is accurate after stopping."""
        sw = StopWatch(precision=2)
        sw.start()
        time.sleep(0.1)
        sw.stop()

        self.assertAlmostEqual(sw.elapsed_time, 0.1, places=1)

    def test_elapsed_time_while_running(self):
        """Test that elapsed_time can be accessed while timer is running."""
        sw = StopWatch(precision=2)
        sw.start()
        time.sleep(0.1)

        # Should be able to check elapsed time while running
        running_time = sw.elapsed_time
        self.assertAlmostEqual(running_time, 0.1, places=1)

        time.sleep(0.1)
        sw.stop()

        # Final time should be ~0.2
        self.assertAlmostEqual(sw.elapsed_time, 0.2, places=1)

    def test_precision_rounding(self):
        """Test that precision parameter correctly rounds values."""
        sw = StopWatch(precision=3)
        sw.start()
        time.sleep(0.123456)
        elapsed = sw.stop()

        # Check that result is rounded to 3 decimal places
        self.assertEqual(len(str(elapsed).split('.')[-1]), 3)

    def test_precision_none(self):
        """Test that precision=None returns full precision."""
        sw = StopWatch(precision=None)
        sw.start()
        time.sleep(0.123456)
        elapsed = sw.stop()

        # Should have high precision (not rounded)
        self.assertAlmostEqual(elapsed, 0.123456, places=2)

    def test_repr_before_start(self):
        """Test __repr__ before timer is started."""
        sw = StopWatch()
        repr_str = repr(sw)

        self.assertIn("StopWatch", repr_str)
        self.assertIn("status=pending", repr_str)

    def test_repr_after_timing(self):
        """Test __repr__ after timer has been used."""
        sw = StopWatch(precision=2)
        sw.start()
        time.sleep(0.1)
        sw.stop()
        repr_str = repr(sw)

        self.assertIn("StopWatch", repr_str)
        self.assertIn("elapsed_time=", repr_str)
        self.assertIn("status=stopped", repr_str)

    def test_context_manager_success(self):
        """Test StopWatch context manager successful execution."""
        with StopWatch(precision=2) as sw:
            time.sleep(0.1)

        self.assertEqual(sw.status, Status.STOPPED)
        self.assertAlmostEqual(sw.elapsed_time, 0.1, places=1)

    def test_context_manager_exception_sets_failed(self):
        """Test StopWatch context manager marks FAILED on exception."""
        sw = None
        with self.assertRaises(ValueError):
            with StopWatch(precision=2) as sw:
                raise ValueError("boom")

        assert sw is not None
        self.assertEqual(sw.status, Status.FAILED)


class TestTimer(unittest.TestCase):
    def test_timer_context_equality_pending(self):
        t1 = TimerContext(precision=2)
        t2 = TimerContext(precision=2)
        self.assertEqual(t1, t2)

        t3 = TimerContext(precision=3)
        self.assertNotEqual(t1, t3)

        with t1:
            time.sleep(0.01)
        self.assertNotEqual(t1, t2)

    def test_registry(self):
        timer = Timer(precision=2)
        with timer['test0']:
            time.sleep(0.1)
        with timer['test1']:
            time.sleep(0.2)
        registry = timer.times
        self.assertIsInstance(registry, dict)
        # type checking ignores for the isinstance check below for testing purposes
        self.assertTrue(all(isinstance(k, str) for k in registry.keys())) # type: ignore
        # After refactoring, times returns list[float] for all values
        self.assertTrue(all(isinstance(v, list) for v in registry.values())) # type: ignore (for testing purposes)
        self.assertTrue(all(isinstance(t, float) for times in registry.values() for t in times))
        self.assertEqual(len(registry), 2)
    
    def test_timer_initialization(self):
        timer = Timer(precision=3)
        self.assertEqual(timer.precision, 3)
        self.assertEqual(timer.times, {})

    def test_timer_context_management(self):
        timer = Timer(precision=2)
        with timer['test'] as t:
            time.sleep(0.1)
        # After refactoring, times['test'] returns a list
        self.assertEqual(len(timer.times['test']), 1)
        self.assertAlmostEqual(timer.times['test'][0], 0.1, places=1)
        self.assertAlmostEqual(t.elapsed_time, 0.1, places=1)

    def test_timer_multiple_contexts(self):
        timer = Timer(precision=2)
        with timer['test1']:
            time.sleep(0.1)
        with timer['test2']:
            time.sleep(0.2)
        # After refactoring, times returns lists
        self.assertEqual(len(timer.times['test1']), 1)
        self.assertEqual(len(timer.times['test2']), 1)
        self.assertAlmostEqual(timer.times['test1'][0], 0.1, places=1)
        self.assertAlmostEqual(timer.times['test2'][0], 0.2, places=1)

    def test_timer_reset(self):
        timer = Timer(precision=2)
        with timer['test'] as t:
            time.sleep(0.1)
            t.reset()
        # After reset within context, elapsed time should be close to 0
        self.assertEqual(len(timer.times['test']), 1)
        self.assertAlmostEqual(timer.times['test'][0], 0.0, places=4)

    def test_timer_repr(self):
        timer = Timer(precision=2)
        with timer['test']:
            time.sleep(0.1)
        self.assertIn("Timer(precision=2, times=", repr(timer))

    def test_timer_no_context(self):
        timer = Timer(precision=2)
        with self.assertRaises(KeyError):
            _ = timer.times['nonexistent']

    def test_timer_overwrite_key(self):
        timer = Timer(precision=2)
        with timer['test']:
            time.sleep(0.1)
        with timer['test']:
            time.sleep(0.2)
        # After refactoring, using the same key twice creates two contexts
        self.assertEqual(len(timer.times['test']), 2)
        self.assertAlmostEqual(timer.times['test'][0], 0.1, places=1)
        self.assertAlmostEqual(timer.times['test'][1], 0.2, places=1)
        
    def test_timer_anonymous_context(self):
        timer = Timer(precision=2)
        with timer.anonymous(name='anonymous_test') as t:
            time.sleep(0.1)
        self.assertAlmostEqual(t.elapsed_time, 0.1, places=1)
        self.assertNotIn('anonymous_test', timer.times)
        self.assertEqual(t.name, 'anonymous_test')
        self.assertEqual(timer.times, {})

    def test_benchmark_store_option(self):
        timer = Timer(precision=3)

        def f():
            time.sleep(0.01)

        # default: benchmark does not persist in registry
        timer.benchmark(f, num_iter=3, warmup=0)
        self.assertNotIn('f benchmark', timer.times)

        # store=True: each iteration is persisted
        timer.benchmark(f, num_iter=4, warmup=0, store=True)
        self.assertIn('f benchmark', timer.times)
        self.assertEqual(len(timer.times['f benchmark']), 4)
        self.assertTrue(all(isinstance(x, float) for x in timer.times['f benchmark']))
        
    def test_sorted_default_order(self):
        timer = Timer(precision=2)
        with timer['short'] as _:
            time.sleep(0.1)
        with timer['medium'] as _:
            time.sleep(0.2)
        with timer['long'] as _:
            time.sleep(0.3)
        
        sorted_timers = timer.sorted()
        self.assertEqual(len(sorted_timers), 3)
        self.assertEqual(sorted_timers[0][1].name, 'short')
        self.assertEqual(sorted_timers[1][1].name, 'medium')
        self.assertEqual(sorted_timers[2][1].name, 'long')

    def test_sorted_reverse_order(self):
        timer = Timer(precision=2)
        with timer['short'] as _:
            time.sleep(0.1)
        with timer['medium'] as _:
            time.sleep(0.2)
        with timer['long'] as _:
            time.sleep(0.3)
        
        sorted_timers = timer.sorted(reverse=True)
        self.assertEqual(len(sorted_timers), 3)
        self.assertEqual(sorted_timers[0][1].name, 'long')
        self.assertEqual(sorted_timers[1][1].name, 'medium')
        self.assertEqual(sorted_timers[2][1].name, 'short')

    def test_sorted_excludes_failed_timers(self):
        timer = Timer(precision=2)
        with timer['success'] as _:
            time.sleep(0.1)
        try:
            with timer['failure']:
                raise Exception("Simulated failure")
        except Exception:
            pass
        
        sorted_timers = timer.sorted()
        self.assertEqual(len(sorted_timers), 1)
        self.assertEqual(sorted_timers[0][1].name, 'success')
        self.assertEqual(sorted_timers[0][1].status, Status.SUCCEEDED)

    def test_sorted_empty_registry(self):
        timer = Timer(precision=2)
        sorted_timers = timer.sorted()
        self.assertEqual(len(sorted_timers), 0)

    def test_status_with_invalid_key(self):
        """Test that status() raises KeyError for nonexistent keys."""
        timer = Timer(precision=2)
        with timer['existing_key']:
            time.sleep(0.05)

        # Valid key should work - returns list after refactoring
        statuses = timer.status('existing_key')
        self.assertIsInstance(statuses, list)
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0], Status.SUCCEEDED)

        # Invalid key should raise KeyError
        with self.assertRaises(KeyError) as context:
            timer.status('nonexistent_key')
        self.assertIn("not found in timer registry", str(context.exception))
        self.assertIn("nonexistent_key", str(context.exception))

    def test_benchmark_not_stored_by_default(self):
        timer = Timer(precision=6)

        def sample() -> None:
            time.sleep(0.001)

        results = timer.benchmark(sample, num_iter=3)

        self.assertEqual(len(results), 3)
        self.assertNotIn('sample benchmark', timer.times)

    def test_benchmark_can_store_results(self):
        timer = Timer(precision=6)

        def sample() -> None:
            time.sleep(0.001)

        results = timer.benchmark(sample, num_iter=3, store=True)

        self.assertEqual(len(results), 3)
        self.assertIn('sample benchmark', timer.times)
        self.assertEqual(len(timer.times['sample benchmark']), 3)
        for elapsed in timer.times['sample benchmark']:
            self.assertGreater(elapsed, 0)

    def test_benchmark_can_store_results_with_custom_key(self):
        timer = Timer(precision=6)

        def sample() -> None:
            time.sleep(0.001)

        results = timer.benchmark(sample, num_iter=2, store=True, key='bench.custom')

        self.assertEqual(len(results), 2)
        self.assertIn('bench.custom', timer.times)
        self.assertNotIn('sample benchmark', timer.times)
        self.assertEqual(len(timer.times['bench.custom']), 2)

    def test_time_call_times_once(self):
        timer = Timer(precision=3)

        def sample() -> int:
            time.sleep(0.01)
            return 42

        ctx = timer.time_call(sample)
        self.assertEqual(ctx.status, Status.SUCCEEDED)
        self.assertIn('sample', timer.times)
        self.assertEqual(len(timer.times['sample']), 1)

    def test_timeit_warns_and_delegates(self):
        timer = Timer(precision=3)

        def sample() -> int:
            return 7

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', DeprecationWarning)
            ctx = timer.timeit(sample)

        self.assertEqual(ctx.status, Status.SUCCEEDED)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertIn('sample', timer.times)


class TestTimerFunctionWrapper(unittest.TestCase):
    def test_time_function_basic(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function():
            time.sleep(0.1)

        sample_function()
        # After refactoring, decorator uses function name as key (not numbered)
        self.assertIn('sample_function', timer.times)
        self.assertEqual(len(timer.times['sample_function']), 1)
        self.assertAlmostEqual(timer.times['sample_function'][0], 0.1, places=1)

    def test_time_function_multiple_calls(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function():
            time.sleep(0.1)

        sample_function()
        sample_function()
        # After refactoring, multiple calls create a list of contexts under one key
        self.assertIn('sample_function', timer.times)
        self.assertEqual(len(timer.times['sample_function']), 2)
        self.assertAlmostEqual(timer.times['sample_function'][0], 0.1, places=1)
        self.assertAlmostEqual(timer.times['sample_function'][1], 0.1, places=1)

    def test_time_function_with_arguments(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function(x: int, y: int) -> int:
            time.sleep(0.1)
            return x + y

        result = sample_function(2, 3)
        self.assertEqual(result, 5)
        # After refactoring, uses function name without numbering
        self.assertIn('sample_function', timer.times)
        self.assertEqual(len(timer.times['sample_function']), 1)
        self.assertAlmostEqual(timer.times['sample_function'][0], 0.1, places=1)

    def test_time_function_different_functions(self):
        timer = Timer(precision=2)

        @timer.timed
        def function_one():
            time.sleep(0.1)

        @timer.timed
        def function_two():
            time.sleep(0.2)

        function_one()
        function_two()
        # After refactoring, uses function names as keys
        self.assertIn('function_one', timer.times)
        self.assertIn('function_two', timer.times)
        self.assertEqual(len(timer.times['function_one']), 1)
        self.assertEqual(len(timer.times['function_two']), 1)
        self.assertAlmostEqual(timer.times['function_one'][0], 0.1, places=1)
        self.assertAlmostEqual(timer.times['function_two'][0], 0.2, places=1)
        
    def test_complex_timer_function(self):
        func_timer = Timer(precision=2)
        @func_timer.timed
        def foo(x: float) -> float:
            time.sleep(x)
            return x
        @func_timer.timed
        def complex_function(x: str) -> Timer:
            inside_timer = Timer(precision=2)
            with inside_timer[x]:
                time.sleep(float(x))
            return inside_timer

        main_timer = Timer(precision=2)
        with main_timer['main']:
            time.sleep(0.1)
            foo(0.1)
            ins_t = complex_function('0.25')
            foo(0.2)
        print(main_timer, main_timer.times)
        print(func_timer, func_timer.times)
        print(ins_t, ins_t.times)
        # After refactoring, keys are function names (not numbered)
        self.assertIn('main', main_timer.times)
        self.assertIn('foo', func_timer.times)
        self.assertIn('complex_function', func_timer.times)
        self.assertIn('0.25', ins_t.times)
        self.assertAlmostEqual(main_timer.times['main'][0], 0.1 + 0.1 + 0.25 + 0.2, places=1)
        # foo was called twice, so it has 2 entries
        self.assertEqual(len(func_timer.times['foo']), 2)
        self.assertAlmostEqual(func_timer.times['foo'][0], 0.1, places=1)
        self.assertAlmostEqual(func_timer.times['foo'][1], 0.2, places=1)
        self.assertAlmostEqual(func_timer.times['complex_function'][0], 0.25, places=1)
        self.assertAlmostEqual(ins_t.times['0.25'][0], 0.25, places=1)
        # Access the most recent context from the list
        self.assertAlmostEqual(ins_t.contexts['0.25'][-1].elapsed_time, 0.25, places=1)      
    
class TestTimerGetMethod(unittest.TestCase):
    def test_fail_get_partial_key(self):
        timer = Timer(precision=2)
        with timer['test_key_1']:
            pass
        with timer['test_key_2']:
            pass
        result = timer.get('test_key')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_get_non_existing_key(self):
        timer = Timer(precision=2)
        with timer['test_key_1']:
            pass
        result = timer.get('nonexistent_key')
        self.assertEqual(result, [])

    def test_get_with_default_value(self):
        timer = Timer(precision=2)
        result = timer.get('nonexistent_key')
        self.assertEqual(result, [])
        
class TestTimerAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.timer = Timer(precision=2)

    async def test_async_function_timing(self):
        @self.timer.timed_async
        async def async_task(delay: float) -> float:
            await asyncio.sleep(delay)
            return delay

        # Call async function 3 times with different delays
        result1 = await async_task(0.2)
        result2 = await async_task(0.4)
        result3 = await async_task(0.6)

        # Check return values
        self.assertEqual(result1, 0.2)
        self.assertEqual(result2, 0.4)
        self.assertEqual(result3, 0.6)

        # After refactoring, all calls go under one key 'async_task'
        times = self.timer.times
        self.assertEqual(len(times), 1)
        self.assertIn('async_task', times)

        # Check that we have 3 timings in the list
        self.assertEqual(len(times['async_task']), 3)

        # Check that all elapsed times are approximately equal to delays
        self.assertAlmostEqual(times['async_task'][0], 0.2, 1)
        self.assertAlmostEqual(times['async_task'][1], 0.4, 1)
        self.assertAlmostEqual(times['async_task'][2], 0.6, 1)

        # Check status and elapsed time via get
        contexts = self.timer.get('async_task')
        self.assertEqual(len(contexts), 3)
        for ctx in contexts:
            self.assertIsInstance(ctx, TimerContext)
            self.assertEqual(ctx.status, Status.SUCCEEDED)
            self.assertGreaterEqual(ctx.elapsed_time, 0.2)

    async def test_async_error_handling(self):
        @self.timer.timed_async
        async def failing_task():
            await asyncio.sleep(0.1)
            raise RuntimeError("Intentional failure")

        with self.assertRaises(RuntimeError):
            await failing_task()

        # Check that status is marked as FAILED
        context = self.timer.get("failing_task")[0]
        self.assertEqual(context.status, Status.FAILED)
        self.assertAlmostEqual(context.elapsed_time, 0.1, 1)
        
if __name__ == "__main__":
    unittest.main()