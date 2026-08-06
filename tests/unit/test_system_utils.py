import unittest
from auto_mobility.utils import benchmark_hw, inspect_system

class TestSystemUtilsUnit(unittest.TestCase):
    def test_benchmark_hw_exports(self):
        self.assertTrue(hasattr(benchmark_hw, "__file__"))

    def test_inspect_system_exports(self):
        self.assertTrue(hasattr(inspect_system, "__file__"))

if __name__ == "__main__":
    unittest.main()
