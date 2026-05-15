import unittest
import os
import shutil
import json
from pathlib import Path
from scripts.utils.integrity import AtomicSingleWriter

class TestIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/tests/scratch_integrity")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.test_dir / "test.lock"
        if self.lock_file.exists():
            os.remove(self.lock_file)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_atomic_lock_prevention(self):
        writer1 = AtomicSingleWriter(self.lock_file, "RUN1", 1234)
        writer2 = AtomicSingleWriter(self.lock_file, "RUN2", 5678)

        # First writer succeeds
        self.assertTrue(writer1.acquire())
        self.assertTrue(self.lock_file.exists())

        # Second writer must fail
        self.assertFalse(writer2.acquire())

        # Verify lock content
        with open(self.lock_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["run_id"], "RUN1")

        # Release writer 1
        writer1.release()
        self.assertFalse(self.lock_file.exists())

        # Now writer 2 can succeed
        self.assertTrue(writer2.acquire())
        self.assertTrue(self.lock_file.exists())
        writer2.release()

    def test_isolation_logic_simulation(self):
        # Simular lo que hacen los runners
        run_id = "TEST_RUN"
        run_dir = self.test_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Escribir algo aislado
        out_file = run_dir / "test_output.csv"
        with open(out_file, "w") as f:
            f.write("data,run_id\n123,TEST_RUN\n")
            
        self.assertTrue(out_file.exists())
        self.assertIn(run_id, out_file.as_posix())

if __name__ == "__main__":
    unittest.main()
