import unittest
import numpy as np
from datetime import datetime
from pytidespy3.astro import astro, AstronomicalParameter
from pytidespy3.constituent import *
from pytidespy3.constituent import (
    _M2, _S2, _N2, _K1, _M4, _O1, _M6, _MK3, _S4, _MN4, _nu2, _S6, _mu2,
    _2N2, _OO1, _lambda2, _S1, _M1, _J1, _Mm, _Ssa, _Sa, _MSF, _Mf,
    _rho1, _Q1, _T2, _R2, _2Q1, _P1, _2SM2, _M3, _L2, _2MK3, _K2,
    _M8, _MS4, noaa
)


class TestBaseConstituent(unittest.TestCase):
    """Test all methods of the BaseConstituent class."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

    def test_init_with_coefficients(self):
        """Test initialization with coefficient list."""
        coeffs = [1, 2, 3, 4, 5, 6, 7]
        const = BaseConstituent("test", coefficients=coeffs)

        self.assertEqual(const.name, "test")
        np.testing.assert_array_equal(const.coefficients, np.array(coeffs))

    def test_init_with_xdo(self):
        """Test initialization with XDO string."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")

        self.assertEqual(const.name, "M2")
        expected_coeffs = np.array([2, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected_coeffs)

    def test_xdo_to_coefficients(self):
        """Test conversion from XDO to coefficients."""
        const = BaseConstituent("test")

        coeffs = const.xdo_to_coefficients("B ZZZ ZZZ")
        expected = [2, 0, 0, 0, 0, 0, 0]
        self.assertEqual(coeffs, expected)

        coeffs = const.xdo_to_coefficients("A AZZ ZZY")
        expected = [1, 1, 0, 0, 0, 0, -1]
        self.assertEqual(coeffs, expected)

    def test_coefficients_to_xdo(self):
        """Test conversion from coefficients to XDO."""
        const = BaseConstituent("test")

        coeffs = [2, 0, 0, 0, 0, 0, 0]
        xdo = const.coefficients_to_xdo(coeffs)
        self.assertEqual(xdo, "B ZZZ ZZZ")

        coeffs = [1, 1, 0, 0, 0, 0, -1]
        xdo = const.coefficients_to_xdo(coeffs)
        self.assertEqual(xdo, "A AZZ ZZY")

    def test_V(self):
        """Test V method."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        V = const.V(self.astro_data)

        self.assertIsInstance(V, (int, float))
        self.assertGreaterEqual(V, 0)
        self.assertLessEqual(V, 360)

    def test_speed(self):
        """Test speed method."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        speed = const.speed(self.astro_data)

        self.assertIsInstance(speed, (int, float))
        self.assertGreater(speed, 0)

    def test_xdo(self):
        """Test xdo method."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        xdo = const.xdo()

        self.assertEqual(xdo, "B ZZZ ZZZ")

    def test_astro_xdo(self):
        """Test astro_xdo method."""
        const = BaseConstituent("test")
        astro_xdo = const.astro_xdo(self.astro_data)

        self.assertEqual(len(astro_xdo), 7)
        for param in astro_xdo:
            self.assertIsInstance(param, AstronomicalParameter)

    def test_astro_speeds(self):
        """Test astro_speeds method."""
        const = BaseConstituent("test")
        speeds = const.astro_speeds(self.astro_data)

        self.assertIsInstance(speeds, np.ndarray)
        self.assertEqual(len(speeds), 7)

    def test_astro_values(self):
        """Test astro_values method."""
        const = BaseConstituent("test")
        values = const.astro_values(self.astro_data)

        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(len(values), 7)

    def test_equality(self):
        """Test equality comparison."""
        const1 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const3 = BaseConstituent("S2", xdo="B BXZ ZZZ")

        self.assertEqual(const1, const2)
        self.assertNotEqual(const1, const3)

    def test_hash(self):
        """Test hash functionality."""
        const1 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")

        self.assertEqual(hash(const1), hash(const2))

    def test_u_and_f_defaults(self):
        """Test default u and f functions."""
        const = BaseConstituent("test", xdo="B ZZZ ZZZ")

        u_value = const.u(self.astro_data)
        f_value = const.f(self.astro_data)

        self.assertIsInstance(u_value, (int, float))
        self.assertIsInstance(f_value, (int, float))
        self.assertGreater(f_value, 0)


class TestCompoundConstituent(unittest.TestCase):
    """Test all methods of the CompoundConstituent class."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

        self.M2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        self.S2 = BaseConstituent("S2", xdo="B BXZ ZZZ")
        self.O1 = BaseConstituent("O1", xdo="A YZZ ZZA")

    def test_init_with_members(self):
        """Test initialization with members."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        self.assertEqual(const.name, "MS4")
        self.assertEqual(len(const.members), 2)

    def test_coefficients_calculation(self):
        """Test coefficient calculation."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        expected = np.array([4, 2, -2, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected)

    def test_speed_calculation(self):
        """Test speed calculation."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        speed = const.speed(self.astro_data)
        m2_speed = self.M2.speed(self.astro_data)
        s2_speed = self.S2.speed(self.astro_data)

        self.assertAlmostEqual(speed, m2_speed + s2_speed, places=10)

    def test_V_calculation(self):
        """Test V calculation."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        V = const.V(self.astro_data)
        m2_V = self.M2.V(self.astro_data)
        s2_V = self.S2.V(self.astro_data)

        expected = (m2_V + s2_V) % 360
        self.assertAlmostEqual(V, expected, places=10)

    def test_u_calculation(self):
        """Test u calculation."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        u = const.u(self.astro_data)
        m2_u = self.M2.u(self.astro_data)
        s2_u = self.S2.u(self.astro_data)

        expected = m2_u + s2_u
        self.assertAlmostEqual(u, expected, places=10)

    def test_f_calculation(self):
        """Test f calculation."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent(members=members, name="MS4")

        f = const.f(self.astro_data)
        m2_f = self.M2.f(self.astro_data)
        s2_f = self.S2.f(self.astro_data)

        expected = m2_f * s2_f
        self.assertAlmostEqual(f, expected, places=10)

    def test_negative_coefficients(self):
        """Test compound constituent with negative coefficients."""
        members = [(self.S2, 2), (self.M2, -1)]
        const = CompoundConstituent(members=members, name="2SM2")

        expected = np.array([2, 4, -4, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected)


class TestConstituentDefinitions(unittest.TestCase):
    """Test defined harmonic constituents."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

    def test_base_constituents_exist(self):
        """Test that base harmonic constituents exist."""
        major_constituents = [_M2, _S2, _K1, _O1, _N2, _P1, _K2]

        for const in major_constituents:
            self.assertIsInstance(const, BaseConstituent)

    def test_compound_constituents_exist(self):
        """Test that compound harmonic constituents exist."""
        compound_constituents = [_MS4, _MN4, _M4, _M6, _2MK3]

        for const in compound_constituents:
            self.assertIsInstance(const, CompoundConstituent)

    def test_noaa_list(self):
        """Test NOAA harmonic constituent list."""
        self.assertIsInstance(noaa, list)
        self.assertGreater(len(noaa), 0)

        for const in noaa:
            self.assertIsInstance(const, (BaseConstituent, CompoundConstituent))

    def test_constituent_properties(self):
        """Test basic properties of harmonic constituents."""
        M2 = _M2
        self.assertEqual(M2.name, "M2")
        self.assertEqual(M2.xdo(), "B ZZZ ZZZ")

        S2 = _S2
        self.assertEqual(S2.name, "S2")
        self.assertEqual(S2.xdo(), "B BXZ ZZZ")

        K1 = _K1
        self.assertEqual(K1.name, "K1")
        self.assertEqual(K1.xdo(), "A AZZ ZZY")

    def test_constituent_speeds(self):
        """Test speeds of harmonic constituents."""
        major_constituents = [
            _M2,
            _S2,
            _K1,
            _O1,
            _N2,
        ]

        for const in major_constituents:
            speed = const.speed(self.astro_data)
            self.assertGreater(speed, 0)

    def test_constituent_V_values(self):
        """Test V values of harmonic constituents."""
        major_constituents = [
            _M2,
            _S2,
            _K1,
            _O1,
            _N2,
        ]

        for const in major_constituents:
            V = const.V(self.astro_data)
            self.assertGreaterEqual(V, 0)
            self.assertLessEqual(V, 360)

    def test_constituent_node_factors(self):
        """Test node factors of harmonic constituents."""
        major_constituents = [
            _M2,
            _S2,
            _K1,
            _O1,
            _N2,
        ]

        for const in major_constituents:
            f = const.f(self.astro_data)
            self.assertGreater(f, 0)

    def test_compound_constituent_calculations(self):
        """Test calculations of compound harmonic constituents."""
        MS4 = _MS4
        M2 = _M2
        S2 = _S2

        expected_coeffs = M2.coefficients + S2.coefficients
        np.testing.assert_array_equal(MS4.coefficients, expected_coeffs)

        expected_speed = M2.speed(self.astro_data) + S2.speed(self.astro_data)
        self.assertAlmostEqual(MS4.speed(self.astro_data), expected_speed, places=10)


if __name__ == "__main__":
    unittest.main()
