import unittest
import numpy as np
from datetime import datetime, timedelta

from pytidespy3.astro import (
    AstronomicalParameter,
    AstronomicalParameters,
    astro,
    d_polynomial,
    polynomial,
    s2d,
    T,
    JD,
)


class TestAstroUtilities(unittest.TestCase):
    """Unit tests for low-level astronomical helpers."""

    def test_s2d(self):
        self.assertAlmostEqual(s2d(30, 15, 45), 30.2625, places=6)
        self.assertAlmostEqual(s2d(0, 30, 0), 0.5, places=6)
        self.assertAlmostEqual(s2d(-30, 15, 45), -29.7375, places=6)

    def test_polynomial(self):
        coeffs = [1, 2, 3]
        self.assertEqual(polynomial(coeffs, 0), 1)
        self.assertEqual(polynomial(coeffs, 1), 6)
        self.assertEqual(polynomial(coeffs, 2), 17)
        self.assertEqual(polynomial([], 5), 0)

    def test_d_polynomial(self):
        coeffs = [1, 2, 3]
        self.assertEqual(d_polynomial(coeffs, 0), 2)
        self.assertEqual(d_polynomial(coeffs, 1), 8)
        self.assertEqual(d_polynomial(coeffs, 2), 14)
        self.assertEqual(d_polynomial([], 5), 0)

    def test_julian_conversions(self):
        epoch = datetime(2000, 1, 1, 12, 0, 0)
        later = datetime(2023, 1, 1, 12, 0, 0)
        self.assertGreater(JD(later), JD(epoch))
        self.assertGreater(T(later), T(epoch))


class TestAstroResults(unittest.TestCase):
    """Golden-data driven tests for astro() outputs."""

    def setUp(self):
        self.single_time = datetime(2023, 1, 1, 12, 0, 0)
        self.multi_times = [
            self.single_time,
            self.single_time + timedelta(hours=12),
            self.single_time + timedelta(hours=24),
        ]
        self.single_result = astro(self.single_time)
        self.multi_result = astro(self.multi_times)

    def test_result_types(self):
        self.assertIsInstance(self.single_result, AstronomicalParameters)
        for value in self.single_result.values():
            self.assertIsInstance(value, AstronomicalParameter)

        self.assertIsInstance(self.multi_result, AstronomicalParameters)
        self.assertEqual(len(self.multi_result.times), len(self.multi_times))

    def test_single_time_golden_values(self):
        expected = {
            "s": (280.88993878027395, 0.041068640165954996),
            "h": (33.22319679267821, 0.5490165192018092),
            "p": (299.25370112060114, 0.004641808076694823),
            "N": (40.18010389581332, -0.0022064056918558415),
            "pp": (283.3327606597261, 1.9612494506754363e-06),
            "90": (90.0, 0.0),
            "omega": (23.436300432425945, 0.0),
            "i": (5.145, 0.0),
            "I": (27.556402814835234, 0.0),
            "xi": (16.844708362082006, 0.0),
            "nu": (3.5923089133099246, 0.0),
            "nup": (18.565687015088628, 0.0),
            "nupp": (66.87491040040767, 0.0),
            "T+h-s": (112.33325801240426, 15.507947879035854),
            "P": (299.25370112060114, 0.004641808076694823),
        }

        for key, (value, speed) in expected.items():
            with self.subTest(key=key):
                param = self.single_result[key]
                self.assertIsInstance(param, AstronomicalParameter)
                self.assertAlmostEqual(param.value, value, places=9)
                self.assertAlmostEqual(param.speed, speed, places=12)

    def test_multi_time_arrays(self):
        expected_values = {
            "s": np.array([280.88993878027395, 281.38276246167856, 281.8755861430832]),
            "h": np.array([33.22319679267821, 39.81139502176493, 46.39959325085165]),
            "T+h-s": np.array([112.33325801240426, 298.428632557704, 124.52400710300373]),
        }
        expected_speed = 0.041068640165954996

        for key, values in expected_values.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(self.multi_result[key].value, values, rtol=0, atol=1e-8)

        np.testing.assert_allclose(
            self.multi_result["s"].speed,
            np.full(3, expected_speed),
            rtol=0,
            atol=1e-12,
        )

    def test_for_time_matches_single(self):
        extracted = self.multi_result.for_time(0)
        self.assertIsInstance(extracted, AstronomicalParameters)
        for key in self.single_result:
            self.assertAlmostEqual(extracted[key].value, self.single_result[key].value, places=9)
            self.assertAlmostEqual(extracted[key].speed, self.single_result[key].speed, places=12)

    def test_t_plus_h_minus_s_identity(self):
        param = self.single_result["T+h-s"]
        computed = ((JD(self.single_time) - int(JD(self.single_time))) * 360.0 +
                    self.single_result["h"].value -
                    self.single_result["s"].value) % 360.0
        self.assertAlmostEqual(param.value, computed, places=9)
        self.assertAlmostEqual(
            param.speed,
            15.0 + self.single_result["h"].speed - self.single_result["s"].speed,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
