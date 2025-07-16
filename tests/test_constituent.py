import unittest
import numpy as np
from datetime import datetime
from pytidespy3.astro import *
from pytidespy3.constituent import *


class TestBaseConstituent(unittest.TestCase):
    """BaseConstituent 클래스의 모든 메서드를 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

    def test_init_with_coefficients(self):
        """계수 리스트로 초기화하는 것을 테스트합니다."""
        coeffs = [1, 2, 3, 4, 5, 6, 7]
        const = BaseConstituent("test", coefficients=coeffs)

        self.assertEqual(const.name, "test")
        np.testing.assert_array_equal(const.coefficients, np.array(coeffs))

    def test_init_with_xdo(self):
        """XDO 문자열로 초기화하는 것을 테스트합니다."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")

        self.assertEqual(const.name, "M2")
        # M2의 계수는 [2, 0, 0, 0, 0, 0, 0]이어야 함
        expected_coeffs = np.array([2, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected_coeffs)

    def test_xdo_to_coefficients(self):
        """XDO를 계수로 변환하는 것을 테스트합니다."""
        const = BaseConstituent("test")

        # M2 성분의 XDO
        coeffs = const.xdo_to_coefficients("B ZZZ ZZZ")
        expected = [2, 0, 0, 0, 0, 0, 0]
        self.assertEqual(coeffs, expected)

        # K1 성분의 XDO
        coeffs = const.xdo_to_coefficients("A AZZ ZZY")
        expected = [1, 1, 0, 0, 0, 0, -1]
        self.assertEqual(coeffs, expected)

    def test_coefficients_to_xdo(self):
        """계수를 XDO로 변환하는 것을 테스트합니다."""
        const = BaseConstituent("test")

        # M2 성분
        coeffs = [2, 0, 0, 0, 0, 0, 0]
        xdo = const.coefficients_to_xdo(coeffs)
        self.assertEqual(xdo, "B ZZZ ZZZ")

        # K1 성분
        coeffs = [1, 1, 0, 0, 0, 0, -1]
        xdo = const.coefficients_to_xdo(coeffs)
        self.assertEqual(xdo, "A AZZ ZZY")

    def test_V(self):
        """V 메서드를 테스트합니다."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        V = const.V(self.astro_data)

        self.assertIsInstance(V, (int, float))
        # V는 각도이므로 0-360 범위에 있어야 함
        self.assertGreaterEqual(V, 0)
        self.assertLessEqual(V, 360)

    def test_speed(self):
        """speed 메서드를 테스트합니다."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        speed = const.speed(self.astro_data)

        self.assertIsInstance(speed, (int, float))
        # M2의 속도는 양수여야 함
        self.assertGreater(speed, 0)

    def test_xdo(self):
        """xdo 메서드를 테스트합니다."""
        const = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        xdo = const.xdo()

        self.assertEqual(xdo, "B ZZZ ZZZ")

    def test_astro_xdo(self):
        """astro_xdo 메서드를 테스트합니다."""
        const = BaseConstituent("test")
        astro_xdo = const.astro_xdo(self.astro_data)

        self.assertEqual(len(astro_xdo), 7)
        for param in astro_xdo:
            self.assertIsInstance(param, astro.AstronomicalParameter)

    def test_astro_speeds(self):
        """astro_speeds 메서드를 테스트합니다."""
        const = BaseConstituent("test")
        speeds = const.astro_speeds(self.astro_data)

        self.assertIsInstance(speeds, np.ndarray)
        self.assertEqual(len(speeds), 7)

    def test_astro_values(self):
        """astro_values 메서드를 테스트합니다."""
        const = BaseConstituent("test")
        values = const.astro_values(self.astro_data)

        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(len(values), 7)

    def test_equality(self):
        """동등성 비교를 테스트합니다."""
        const1 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const3 = BaseConstituent("S2", xdo="B BXZ ZZZ")

        self.assertEqual(const1, const2)
        self.assertNotEqual(const1, const3)

    def test_hash(self):
        """해시 기능을 테스트합니다."""
        const1 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        const2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")

        self.assertEqual(hash(const1), hash(const2))

    def test_u_and_f_defaults(self):
        """기본 u와 f 함수를 테스트합니다."""
        const = BaseConstituent("test", xdo="B ZZZ ZZZ")

        u_value = const.u(self.astro_data)
        f_value = const.f(self.astro_data)

        self.assertIsInstance(u_value, (int, float))
        self.assertIsInstance(f_value, (int, float))
        self.assertGreater(f_value, 0)  # f는 항상 양수여야 함


class TestCompoundConstituent(unittest.TestCase):
    """CompoundConstituent 클래스의 모든 메서드를 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

        # 기본 성분들 생성
        self.M2 = BaseConstituent("M2", xdo="B ZZZ ZZZ")
        self.S2 = BaseConstituent("S2", xdo="B BXZ ZZZ")
        self.O1 = BaseConstituent("O1", xdo="A YZZ ZZA")

    def test_init_with_members(self):
        """멤버로 초기화하는 것을 테스트합니다."""
        # MS4 = M2 + S2
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        self.assertEqual(const.name, "MS4")
        self.assertEqual(len(const.members), 2)

    def test_coefficients_calculation(self):
        """계수 계산을 테스트합니다."""
        # MS4 = M2 + S2
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        # M2: [2, 0, 0, 0, 0, 0, 0], S2: [2, 2, 0, 0, 0, 0, 0]
        # MS4: [4, 2, 0, 0, 0, 0, 0]
        expected = np.array([4, 2, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected)

    def test_speed_calculation(self):
        """속도 계산을 테스트합니다."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        speed = const.speed(self.astro_data)
        m2_speed = self.M2.speed(self.astro_data)
        s2_speed = self.S2.speed(self.astro_data)

        self.assertAlmostEqual(speed, m2_speed + s2_speed, places=10)

    def test_V_calculation(self):
        """V 계산을 테스트합니다."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        V = const.V(self.astro_data)
        m2_V = self.M2.V(self.astro_data)
        s2_V = self.S2.V(self.astro_data)

        expected = (m2_V + s2_V) % 360
        self.assertAlmostEqual(V, expected, places=10)

    def test_u_calculation(self):
        """u 계산을 테스트합니다."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        u = const.u(self.astro_data)
        m2_u = self.M2.u(self.astro_data)
        s2_u = self.S2.u(self.astro_data)

        expected = m2_u + s2_u
        self.assertAlmostEqual(u, expected, places=10)

    def test_f_calculation(self):
        """f 계산을 테스트합니다."""
        members = [(self.M2, 1), (self.S2, 1)]
        const = CompoundConstituent("MS4", members=members)

        f = const.f(self.astro_data)
        m2_f = self.M2.f(self.astro_data)
        s2_f = self.S2.f(self.astro_data)

        expected = m2_f * s2_f
        self.assertAlmostEqual(f, expected, places=10)

    def test_negative_coefficients(self):
        """음수 계수를 가진 복합 성분을 테스트합니다."""
        # 2SM2 = 2*S2 - M2
        members = [(self.S2, 2), (self.M2, -1)]
        const = CompoundConstituent("2SM2", members=members)

        # S2: [2, 2, 0, 0, 0, 0, 0], M2: [2, 0, 0, 0, 0, 0, 0]
        # 2SM2: [4, 4, 0, 0, 0, 0, 0] - [2, 0, 0, 0, 0, 0, 0] = [2, 4, 0, 0, 0, 0, 0]
        expected = np.array([2, 4, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(const.coefficients, expected)


class TestConstituentDefinitions(unittest.TestCase):
    """정의된 조화분조들을 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = astro(self.test_time)

    def test_base_constituents_exist(self):
        """기본 조화분조들이 존재하는지 테스트합니다."""
        # 주요 조화분조들 확인
        major_constituents = ["_M2", "_S2", "_K1", "_O1", "_N2", "_P1", "_K2"]

        for const_name in major_constituents:
            self.assertTrue(hasattr(constituent, const_name))
            const = getattr(constituent, const_name)
            self.assertIsInstance(const, BaseConstituent)

    def test_compound_constituents_exist(self):
        """복합 조화분조들이 존재하는지 테스트합니다."""
        # 주요 복합 조화분조들 확인
        compound_constituents = ["_MS4", "_MN4", "_M4", "_M6", "_2MK3"]

        for const_name in compound_constituents:
            self.assertTrue(hasattr(constituent, const_name))
            const = getattr(constituent, const_name)
            self.assertIsInstance(const, CompoundConstituent)

    def test_noaa_list(self):
        """NOAA 조화분조 리스트를 테스트합니다."""
        self.assertIsInstance(noaa, list)
        self.assertGreater(len(noaa), 0)

        # 모든 NOAA 조화분조가 유효한지 확인
        for const in noaa:
            self.assertIsInstance(const, (BaseConstituent, CompoundConstituent))

    def test_constituent_properties(self):
        """조화분조들의 기본 속성들을 테스트합니다."""
        # M2 조화분조 테스트
        M2 = _M2
        self.assertEqual(M2.name, "M2")
        self.assertEqual(M2.xdo(), "B ZZZ ZZZ")

        # S2 조화분조 테스트
        S2 = _S2
        self.assertEqual(S2.name, "S2")
        self.assertEqual(S2.xdo(), "B BXZ ZZZ")

        # K1 조화분조 테스트
        K1 = _K1
        self.assertEqual(K1.name, "K1")
        self.assertEqual(K1.xdo(), "A AZZ ZZY")

    def test_constituent_speeds(self):
        """조화분조들의 속도를 테스트합니다."""
        # 주요 조화분조들의 속도가 양수인지 확인
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
        """조화분조들의 V 값을 테스트합니다."""
        # V 값이 0-360 범위에 있는지 확인
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
        """조화분조들의 노드 인자를 테스트합니다."""
        # f 값이 양수인지 확인
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
        """복합 조화분조들의 계산을 테스트합니다."""
        # MS4 = M2 + S2
        MS4 = _MS4
        M2 = _M2
        S2 = _S2

        # 계수 확인
        expected_coeffs = M2.coefficients + S2.coefficients
        np.testing.assert_array_equal(MS4.coefficients, expected_coeffs)

        # 속도 확인
        expected_speed = M2.speed(self.astro_data) + S2.speed(self.astro_data)
        self.assertAlmostEqual(MS4.speed(self.astro_data), expected_speed, places=10)


if __name__ == "__main__":
    unittest.main()
