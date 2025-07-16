import unittest
import numpy as np
from datetime import datetime
import pytidespy3.nodal_corrections as nc
import pytidespy3.astro


class TestNodalCorrections(unittest.TestCase):
    """Test all functions in the nodal_corrections module."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = pytidespy3.astro.astro(self.test_time)

    def test_f_unity(self):
        """Test f_unity function."""
        result = nc.f_unity(self.astro_data)
        self.assertEqual(result, 1.0)

    def test_f_Mm(self):
        """Test f_Mm function."""
        result = nc.f_Mm(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_Mf(self):
        """Test f_Mf function."""
        result = nc.f_Mf(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_O1(self):
        """Test f_O1 function."""
        result = nc.f_O1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_J1(self):
        """Test f_J1 function."""
        result = nc.f_J1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_OO1(self):
        """Test f_OO1 function."""
        result = nc.f_OO1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_M2(self):
        """Test f_M2 function."""
        result = nc.f_M2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_K1(self):
        """Test f_K1 function."""
        result = nc.f_K1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_L2(self):
        """Test f_L2 function."""
        result = nc.f_L2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_K2(self):
        """Test f_K2 function."""
        result = nc.f_K2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_M1(self):
        """Test f_M1 function."""
        result = nc.f_M1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_Modd(self):
        """Test f_Modd function."""
        result = nc.f_Modd(self.astro_data, 3)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

        result2 = nc.f_Modd(self.astro_data, 5)
        self.assertIsInstance(result2, float)
        self.assertGreater(result2, 0)

    def test_u_zero(self):
        """Test u_zero function."""
        result = nc.u_zero(self.astro_data)
        self.assertEqual(result, 0.0)

    def test_u_Mf(self):
        """Test u_Mf function."""
        result = nc.u_Mf(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_O1(self):
        """Test u_O1 function."""
        result = nc.u_O1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_J1(self):
        """Test u_J1 function."""
        result = nc.u_J1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_OO1(self):
        """Test u_OO1 function."""
        result = nc.u_OO1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_M2(self):
        """Test u_M2 function."""
        result = nc.u_M2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_K1(self):
        """Test u_K1 function."""
        result = nc.u_K1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_L2(self):
        """Test u_L2 function."""
        result = nc.u_L2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_K2(self):
        """Test u_K2 function."""
        result = nc.u_K2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_M1(self):
        """Test u_M1 function."""
        result = nc.u_M1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

    def test_u_Modd(self):
        """Test u_Modd function."""
        result = nc.u_Modd(self.astro_data, 3)

        self.assertIsInstance(result, float)
        self.assertIsInstance(result, (int, float))

        result2 = nc.u_Modd(self.astro_data, 5)
        self.assertIsInstance(result2, float)
        self.assertIsInstance(result2, (int, float))

    def test_node_factors_consistency(self):
        """Test consistency of node factors."""
        f_M2_1 = nc.f_M2(self.astro_data)
        f_M2_2 = nc.f_M2(self.astro_data)

        self.assertAlmostEqual(f_M2_1, f_M2_2, places=10)

    def test_node_factors_time_dependence(self):
        """Test time dependence of node factors."""
        time1 = datetime(2023, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 6, 15, 12, 0, 0)

        astro1 = pytidespy3.astro.astro(time1)
        astro2 = pytidespy3.astro.astro(time2)

        f_M2_1 = nc.f_M2(astro1)
        f_M2_2 = nc.f_M2(astro2)

        self.assertNotAlmostEqual(f_M2_1, f_M2_2, places=5)

    def test_node_factors_ranges(self):
        """Test ranges of node factors."""
        f_functions = [
            nc.f_Mm,
            nc.f_Mf,
            nc.f_O1,
            nc.f_J1,
            nc.f_OO1,
            nc.f_M2,
            nc.f_K1,
            nc.f_L2,
            nc.f_K2,
            nc.f_M1,
        ]

        for f_func in f_functions:
            result = f_func(self.astro_data)
            self.assertGreater(result, 0)
            self.assertLess(result, 2.0)

    def test_u_values_ranges(self):
        """Test ranges of u values."""
        u_functions = [
            nc.u_Mf,
            nc.u_O1,
            nc.u_J1,
            nc.u_OO1,
            nc.u_M2,
            nc.u_K1,
            nc.u_L2,
            nc.u_K2,
            nc.u_M1,
        ]

        for u_func in u_functions:
            result = u_func(self.astro_data)
            self.assertGreater(result, -200)
            self.assertLess(result, 200)

    def test_f_Modd_relationship(self):
        """Test relationship between f_Modd and f_M2."""
        f_M2 = nc.f_M2(self.astro_data)
        f_M3 = nc.f_Modd(self.astro_data, 3)
        f_M5 = nc.f_Modd(self.astro_data, 5)

        expected_f_M3 = f_M2 ** (3 / 2)
        expected_f_M5 = f_M2 ** (5 / 2)

        self.assertAlmostEqual(f_M3, expected_f_M3, places=10)
        self.assertAlmostEqual(f_M5, expected_f_M5, places=10)

    def test_u_Modd_relationship(self):
        """Test relationship between u_Modd and u_M2."""
        u_M2 = nc.u_M2(self.astro_data)
        u_M3 = nc.u_Modd(self.astro_data, 3)
        u_M5 = nc.u_Modd(self.astro_data, 5)

        expected_u_M3 = 3 / 2 * u_M2
        expected_u_M5 = 5 / 2 * u_M2

        self.assertAlmostEqual(u_M3, expected_u_M3, places=10)
        self.assertAlmostEqual(u_M5, expected_u_M5, places=10)

    def test_astro_parameter_usage(self):
        """Test that astro parameters are used correctly."""
        required_keys = ["omega", "i", "I", "xi", "nu", "nup", "nupp", "P"]

        for key in required_keys:
            self.assertIn(key, self.astro_data)

    def test_mathematical_consistency(self):
        """Test mathematical consistency."""
        f_M2 = nc.f_M2(self.astro_data)
        f_L2 = nc.f_L2(self.astro_data)

        self.assertGreaterEqual(f_L2, f_M2 * 0.5)

    def test_edge_cases(self):
        """Test edge cases."""
        past_time = datetime(1900, 1, 1, 12, 0, 0)
        past_astro = pytidespy3.astro.astro(past_time)

        future_time = datetime(2100, 1, 1, 12, 0, 0)
        future_astro = pytidespy3.astro.astro(future_time)

        f_functions = [
            nc.f_Mm,
            nc.f_Mf,
            nc.f_O1,
            nc.f_J1,
            nc.f_OO1,
            nc.f_M2,
            nc.f_K1,
            nc.f_L2,
            nc.f_K2,
            nc.f_M1,
        ]

        for f_func in f_functions:
            past_result = f_func(past_astro)
            future_result = f_func(future_astro)

            self.assertGreater(past_result, 0)
            self.assertGreater(future_result, 0)

    def test_precision(self):
        """Test precision."""
        f_M2_results = []
        u_M2_results = []

        for _ in range(10):
            f_M2_results.append(nc.f_M2(self.astro_data))
            u_M2_results.append(nc.u_M2(self.astro_data))

        self.assertEqual(len(set(f_M2_results)), 1)
        self.assertEqual(len(set(u_M2_results)), 1)


if __name__ == "__main__":
    unittest.main()
