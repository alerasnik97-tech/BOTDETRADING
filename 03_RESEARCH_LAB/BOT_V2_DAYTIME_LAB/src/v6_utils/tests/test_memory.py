
import unittest
import sys
import os

# Asegurar que el path incluya v6_utils
sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

from v6_utils.memory import get_process_rss_mb, MemoryGuard, safe_collect

class TestMemory(unittest.TestCase):
    def test_rss_returns_positive(self):
        rss = get_process_rss_mb()
        self.assertFalse(rss != rss) # check for nan
        self.assertGreater(rss, 0)

    def test_memory_guard_logs_delta(self):
        with MemoryGuard(budget_mb=10000, label="TestDelta") as guard:
            large_list = [i for i in range(1000000)]
            guard.check()
            self.assertGreater(guard.peak_rss, guard.initial_rss)
            del large_list

    def test_memory_guard_aborts_on_breach(self):
        # Budget muy pequeño para forzar breach
        with self.assertRaises(MemoryError):
            with MemoryGuard(budget_mb=1, label="TestBreach") as guard:
                large_list = [i for i in range(1000000)]
                guard.check()

    def test_safe_collect_returns_stats(self):
        stats = safe_collect()
        self.assertIn("n_collected", stats)
        self.assertIn("freed_mb", stats)

if __name__ == '__main__':
    unittest.main()
