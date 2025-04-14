import time
import unittest
from timer import TimerContext, Timer, Status
import asyncio

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
        with self.assertRaises(ValueError):
            timer.elapsed_time = -1

    def test_repr(self):
        with TimerContext(precision=2) as timer:
            time.sleep(0.1)
        self.assertIn("TimerContext(name=Unnamed, elapsed_time=", repr(timer))

    def test_precision_none(self):
        with TimerContext(precision=None) as timer:
            time.sleep(0.123456)
        self.assertAlmostEqual(timer.elapsed_time, 0.123456, places=2)
        
    
class TestTimer(unittest.TestCase):
    def test_registry(self):
        timer = Timer(precision=2)
        with timer['test0']:
            time.sleep(0.1)
        with timer['test1']:
            time.sleep(0.2)
        registry = timer.times
        self.assertIsInstance(registry, dict)
        self.assertTrue(all(isinstance(k, str) for k in registry.keys()))
        self.assertTrue(all(isinstance(v, float) for v in registry.values()))
        self.assertEqual(len(registry), 2)
    
    def test_timer_initialization(self):
        timer = Timer(precision=3)
        self.assertEqual(timer.precision, 3)
        self.assertEqual(timer.times, {})

    def test_timer_context_management(self):
        timer = Timer(precision=2)
        with timer['test'] as t:
            time.sleep(0.1)
        self.assertAlmostEqual(timer.times['test'], 0.1, places=1)
        self.assertAlmostEqual(t.elapsed_time, 0.1, places=1)

    def test_timer_multiple_contexts(self):
        timer = Timer(precision=2)
        with timer['test1']:
            time.sleep(0.1)
        with timer['test2']:
            time.sleep(0.2)
        self.assertAlmostEqual(timer.times['test1'], 0.1, places=1)
        self.assertAlmostEqual(timer.times['test2'], 0.2, places=1)

    def test_timer_reset(self):
        timer = Timer(precision=2)
        with timer['test'] as t:
            time.sleep(0.1)
            t.reset()
        self.assertEqual(timer.times, {'test': 0.0})

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
        self.assertAlmostEqual(timer.times['test'], 0.2, places=1)
        
    def test_timer_anonymous_context(self):
        timer = Timer(precision=2)
        with timer.anonymous(name='anonymous_test') as t:
            time.sleep(0.1)
        self.assertAlmostEqual(t.elapsed_time, 0.1, places=1)
        self.assertNotIn('anonymous_test', timer.times)
        self.assertEqual(t.name, 'anonymous_test')
        self.assertEqual(timer.times, {})
        
    def test_sorted_default_order(self):
        timer = Timer(precision=2)
        with timer['short'] as t:
            time.sleep(0.1)
        with timer['medium'] as t:
            time.sleep(0.2)
        with timer['long'] as t:
            time.sleep(0.3)
        
        sorted_timers = timer.sorted()
        self.assertEqual(len(sorted_timers), 3)
        self.assertEqual(sorted_timers[0][1].name, 'short')
        self.assertEqual(sorted_timers[1][1].name, 'medium')
        self.assertEqual(sorted_timers[2][1].name, 'long')

    def test_sorted_reverse_order(self):
        timer = Timer(precision=2)
        with timer['short'] as t:
            time.sleep(0.1)
        with timer['medium'] as t:
            time.sleep(0.2)
        with timer['long'] as t:
            time.sleep(0.3)
        
        sorted_timers = timer.sorted(reverse=True)
        self.assertEqual(len(sorted_timers), 3)
        self.assertEqual(sorted_timers[0][1].name, 'long')
        self.assertEqual(sorted_timers[1][1].name, 'medium')
        self.assertEqual(sorted_timers[2][1].name, 'short')

    def test_sorted_excludes_failed_timers(self):
        timer = Timer(precision=2)
        with timer['success'] as t:
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

class TestTimerFunctionWrapper(unittest.TestCase):
    def test_time_function_basic(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function():
            time.sleep(0.1)

        sample_function()
        self.assertIn('sample_function (0)', timer.times)
        self.assertAlmostEqual(timer.times['sample_function (0)'], 0.1, places=1)

    def test_time_function_multiple_calls(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function():
            time.sleep(0.1)

        sample_function()
        sample_function()
        self.assertIn('sample_function (0)', timer.times)
        self.assertIn('sample_function (1)', timer.times)
        self.assertAlmostEqual(timer.times['sample_function (0)'], 0.1, places=1)
        self.assertAlmostEqual(timer.times['sample_function (1)'], 0.1, places=1)

    def test_time_function_with_arguments(self):
        timer = Timer(precision=2)

        @timer.timed
        def sample_function(x, y):
            time.sleep(0.1)
            return x + y

        result = sample_function(2, 3)
        self.assertEqual(result, 5)
        self.assertIn('sample_function (0)', timer.times)
        self.assertAlmostEqual(timer.times['sample_function (0)'], 0.1, places=1)

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
        self.assertIn('function_one (0)', timer.times)
        self.assertIn('function_two (0)', timer.times)
        self.assertAlmostEqual(timer.times['function_one (0)'], 0.1, places=1)
        self.assertAlmostEqual(timer.times['function_two (0)'], 0.2, places=1)
        
    def test_complex_timer_function(self):
        func_timer = Timer(precision=2)
        @func_timer.timed
        def foo(x: int) -> int:
            time.sleep(x)
            return x
        @func_timer.timed
        def complex_function(x: int) -> Timer:
            inside_timer = Timer(precision=2)
            with inside_timer[x]:
                time.sleep(x)
            return inside_timer
        
        main_timer = Timer(precision=2)
        with main_timer['main']:
            time.sleep(0.1)
            foo(0.1)
            ins_t = complex_function(0.25)
            foo(0.2)
        print(main_timer, main_timer.times)
        print(func_timer, func_timer.times)
        print(ins_t, ins_t.times)
        self.assertIn('main', main_timer.times)
        self.assertIn('foo (0)', func_timer.times)
        self.assertIn('complex_function (0)', func_timer.times)
        self.assertIn('0.25', ins_t.times)
        self.assertAlmostEqual(main_timer.times['main'], 0.1 + 0.1 + 0.25 + 0.2, places=1)
        self.assertAlmostEqual(func_timer.times['foo (0)'], 0.1, places=1)
        self.assertAlmostEqual(func_timer.times['complex_function (0)'], 0.25, places=1)
        self.assertAlmostEqual(ins_t.times['0.25'], 0.25, places=1)
        self.assertAlmostEqual(ins_t[0.25].elapsed_time, 0.25, places=1)
        self.assertAlmostEqual(func_timer.times['foo (1)'], 0.2, places=1)      
    
class TestTimerGetMethod(unittest.TestCase):
    def test_get_existing_partial_key(self):
        timer = Timer(precision=2)
        with timer['test_key_1']:
            pass
        with timer['test_key_2']:
            pass
        result = timer.get('test_key')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(context._name.startswith('test_key') for context in result))

    def test_get_non_existing_partial_key(self):
        timer = Timer(precision=2)
        with timer['test_key_1']:
            pass
        result = timer.get('nonexistent_key')
        self.assertEqual(result, [])

    def test_get_with_default_value(self):
        timer = Timer(precision=2)
        result = timer.get('nonexistent_key')
        self.assertEqual(result, [])

    def test_get_partial_key_with_multiple_matches(self):
        timer = Timer(precision=2)
        with timer['key_1']:
            pass
        with timer['key_2']:
            pass
        with timer['key_3']:
            pass
        result = timer.get('key')
        self.assertEqual(len(result), 3)
        self.assertTrue(all(context.name.startswith('key') for context in result))
        
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

        # Check that 3 keys were added to timer
        times = self.timer.times
        self.assertEqual(len(times), 3)

        # Check that keys are named correctly
        expected_keys = ['async_task (0)', 'async_task (1)', 'async_task (2)']
        for key in expected_keys:
            self.assertIn(key, times)

        # Check that all elapsed times are greater than or equal to the delay
        for key, delay in zip(expected_keys, [0.2, 0.4, 0.6]):
            self.assertAlmostEqual(times[key], delay, 1)

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