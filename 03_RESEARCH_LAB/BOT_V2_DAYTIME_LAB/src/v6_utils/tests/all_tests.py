
import unittest
import sys
import os

# Asegurar que el path incluya v6_utils
sys.path.append(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

# Importar todos los módulos de test
from v6_utils.tests.test_runner import TestTemporal, TestNumeric # V6.1
from v6_utils.tests.test_memory import TestMemory
from v6_utils.tests.test_data_loader import TestDataLoader
from v6_utils.tests.test_bars import TestBars
from v6_utils.tests.test_causal import TestCausal
from v6_utils.tests.test_execution import TestExecution

if __name__ == '__main__':
    # Crear suite combinada
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTemporal))
    suite.addTests(loader.loadTestsFromTestCase(TestNumeric))
    suite.addTests(loader.loadTestsFromTestCase(TestMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestBars))
    suite.addTests(loader.loadTestsFromTestCase(TestCausal))
    suite.addTests(loader.loadTestsFromTestCase(TestExecution))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)
