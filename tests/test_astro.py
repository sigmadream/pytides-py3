import unittest
import numpy as np
from datetime import datetime, timedelta
from pytidespy3.astro import *


class TestAstro(unittest.TestCase):
    """Test all functions in the astro module."""

    def setUp(self):
        """Set up basic test times."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.test_time2 = datetime(2023, 6, 15, 6, 30, 0)

    def test_s2d_basic(self):
        """Test basic functionality of s2d function."""
        self.assertAlmostEqual(s2d(30, 15, 45), 30.2625, places=6)
        self.assertAlmostEqual(s2d(0, 30, 0), 0.5, places=6)
        self.assertAlmostEqual(s2d(45), 45.0, places=6)

    def test_s2d_edge_cases(self):
        """Test edge cases of s2d function."""
        self.assertEqual(s2d(0, 0, 0), 0.0)
        self.assertAlmostEqual(s2d(-30, 15, 45), -29.7375, places=6)

    def test_polynomial_basic(self):
        """Test basic functionality of polynomial function."""
        coeffs = [1, 2, 3]
        self.assertEqual(polynomial(coeffs, 0), 1)
        self.assertEqual(polynomial(coeffs, 1), 6)
        self.assertEqual(polynomial(coeffs, 2), 17)

    def test_polynomial_empty(self):
        """Test polynomial function with empty coefficient list."""
        self.assertEqual(polynomial([], 5), 0)

    def test_d_polynomial_basic(self):
        """Test basic functionality of d_polynomial function."""
        coeffs = [1, 2, 3]
        self.assertEqual(d_polynomial(coeffs, 0), 2)
        self.assertEqual(d_polynomial(coeffs, 1), 8)
        self.assertEqual(d_polynomial(coeffs, 2), 14)

    def test_d_polynomial_empty(self):
        """Test d_polynomial function with empty coefficient list."""
        self.assertEqual(d_polynomial([], 5), 0)

    def test_T_basic(self):
        """Test basic functionality of T function."""
        result = T(self.test_time)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_T_different_times(self):
        """Test T function with different times."""
        time1 = datetime(2000, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 1, 1, 12, 0, 0)

        T1 = T(time1)
        T2 = T(time2)

        self.assertGreater(T2, T1)

    def test_JD_basic(self):
        """Test basic functionality of JD function."""
        result = JD(self.test_time)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 2400000)

    def test_JD_edge_cases(self):
        """Test edge cases of JD function."""
        jan_date = datetime(2023, 1, 15, 12, 0, 0)
        feb_date = datetime(2023, 2, 15, 12, 0, 0)

        JD_jan = JD(jan_date)
        JD_feb = JD(feb_date)

        self.assertGreater(JD_feb, JD_jan)

    def test_astro_basic(self):
        """Test basic functionality of astro function."""
        result = astro(self.test_time)

        self.assertIsInstance(result, dict)

        required_keys = [
            "s",
            "h",
            "p",
            "N",
            "pp",
            "90",
            "omega",
            "i",
            "I",
            "xi",
            "nu",
            "nup",
            "nupp",
            "T+h-s",
            "P",
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_astro_parameter_structure(self):
        """Test the structure of parameters returned by astro function."""
        result = astro(self.test_time)

        for key, param in result.items():
            if key not in [
                "I",
                "xi",
                "nu",
                "nup",
                "nupp",
                "P",
            ]:
                self.assertIsInstance(param, AstronomicalParameter)
                self.assertIsInstance(param.value, (int, float))
                if param.speed is not None:
                    self.assertIsInstance(param.speed, (int, float))

    def test_astro_value_ranges(self):
        """Test that astro function returns values in expected ranges."""
        result = astro(self.test_time)

        for key, param in result.items():
            if hasattr(param, "value"):
                self.assertGreaterEqual(param.value, 0)
                self.assertLessEqual(param.value, 360)

    def test_astro_consistency(self):
        """Test consistency of astro function."""
        result1 = astro(self.test_time)
        result2 = astro(self.test_time)

        for key in result1:
            if hasattr(result1[key], "value"):
                self.assertAlmostEqual(
                    result1[key].value, result2[key].value, places=10
                )

    def test_astro_time_progression(self):
        """Test astro function behavior with time progression."""
        time1 = datetime(2023, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 1, 1, 13, 0, 0)

        result1 = astro(time1)
        result2 = astro(time2)

        self.assertNotAlmostEqual(
            result1["T+h-s"].value, result2["T+h-s"].value, places=5
        )

    def test_astro_speed_values(self):
        """Test speed values returned by astro function."""
        result = astro(self.test_time)

        self.assertAlmostEqual(result["s"].speed, 0.041, places=3)
        self.assertAlmostEqual(result["h"].speed, 0.549, places=3)

    def test_astro_parameter_relationships(self):
        """Test relationships between parameters returned by astro function."""
        result = astro(self.test_time)

        T_plus_h_minus_s = result["T+h-s"].value
        T_value = (JD(self.test_time) - int(JD(self.test_time))) * 360.0
        h_value = result["h"].value
        s_value = result["s"].value

        expected = (T_value + h_value - s_value) % 360.0
        self.assertAlmostEqual(T_plus_h_minus_s, expected, places=5)

    def test_astro_different_years(self):
        """Test astro function behavior with different years."""
        time_2000 = datetime(2000, 1, 1, 12, 0, 0)
        time_2023 = datetime(2023, 1, 1, 12, 0, 0)

        result_2000 = astro(time_2000)
        result_2023 = astro(time_2023)

        self.assertNotAlmostEqual(
            result_2000["h"].value, result_2023["h"].value, places=5
        )

    def test_astro_microsecond_precision(self):
        """Test astro function behavior with microsecond precision."""
        time1 = datetime(2023, 1, 1, 12, 0, 0, 0)
        time2 = datetime(2023, 1, 1, 12, 0, 0, 100000)

        result1 = astro(time1)
        result2 = astro(time2)

        self.assertNotEqual(result1["T+h-s"].value, result2["T+h-s"].value)


if __name__ == "__main__":
    unittest.main()
