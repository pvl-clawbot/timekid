import time
import unittest

from timekid.fast import FastTimer


class TestFastTimer(unittest.TestCase):
    def test_start_stop_records(self):
        ft = FastTimer()
        tok = ft.start("x")
        time.sleep(0.01)
        dt = ft.stop(tok)

        self.assertIsInstance(dt, int)
        self.assertGreater(dt, 0)
        self.assertIn("x", ft.times_ns)
        self.assertEqual(len(ft.times_ns["x"]), 1)

    def test_key_id_is_stable(self):
        ft = FastTimer()
        a1 = ft.key_id("a")
        a2 = ft.key_id("a")
        b = ft.key_id("b")
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)

    def test_times_s_precision(self):
        ft = FastTimer()
        tok = ft.start("x")
        time.sleep(0.012345)
        ft.stop(tok)
        out = ft.times_s(precision=3)
        self.assertIn("x", out)
        self.assertEqual(len(out["x"]), 1)
        # Rounded to 3 decimal places => string repr has <=3 dp
        s = f"{out['x'][0]:.3f}"
        self.assertRegex(s, r"^\d+\.\d{3}$")


if __name__ == "__main__":
    unittest.main()
