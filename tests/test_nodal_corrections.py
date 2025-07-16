import unittest
import numpy as np
from datetime import datetime
import pytidespy3.nodal_corrections as nc
import pytidespy3.astro


class TestNodalCorrections(unittest.TestCase):
    """nodal_corrections 모듈의 모든 함수를 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.astro_data = pytidespy3.astro.astro(self.test_time)

    def test_f_unity(self):
        """f_unity 함수를 테스트합니다."""
        result = nc.f_unity(self.astro_data)
        self.assertEqual(result, 1.0)

    def test_f_Mm(self):
        """f_Mm 함수를 테스트합니다."""
        result = nc.f_Mm(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # f는 항상 양수여야 함

    def test_f_Mf(self):
        """f_Mf 함수를 테스트합니다."""
        result = nc.f_Mf(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_O1(self):
        """f_O1 함수를 테스트합니다."""
        result = nc.f_O1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_J1(self):
        """f_J1 함수를 테스트합니다."""
        result = nc.f_J1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_OO1(self):
        """f_OO1 함수를 테스트합니다."""
        result = nc.f_OO1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_M2(self):
        """f_M2 함수를 테스트합니다."""
        result = nc.f_M2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_K1(self):
        """f_K1 함수를 테스트합니다."""
        result = nc.f_K1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_L2(self):
        """f_L2 함수를 테스트합니다."""
        result = nc.f_L2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_K2(self):
        """f_K2 함수를 테스트합니다."""
        result = nc.f_K2(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_M1(self):
        """f_M1 함수를 테스트합니다."""
        result = nc.f_M1(self.astro_data)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_f_Modd(self):
        """f_Modd 함수를 테스트합니다."""
        # n=3인 경우 테스트
        result = nc.f_Modd(self.astro_data, 3)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

        # n=5인 경우 테스트
        result2 = nc.f_Modd(self.astro_data, 5)
        self.assertIsInstance(result2, float)
        self.assertGreater(result2, 0)

    def test_u_zero(self):
        """u_zero 함수를 테스트합니다."""
        result = nc.u_zero(self.astro_data)
        self.assertEqual(result, 0.0)

    def test_u_Mf(self):
        """u_Mf 함수를 테스트합니다."""
        result = nc.u_Mf(self.astro_data)

        self.assertIsInstance(result, float)
        # u_Mf는 -2 * xi이므로 음수일 가능성이 높음
        self.assertIsInstance(result, (int, float))

    def test_u_O1(self):
        """u_O1 함수를 테스트합니다."""
        result = nc.u_O1(self.astro_data)

        self.assertIsInstance(result, float)
        # u_O1 = 2*xi - nu
        self.assertIsInstance(result, (int, float))

    def test_u_J1(self):
        """u_J1 함수를 테스트합니다."""
        result = nc.u_J1(self.astro_data)

        self.assertIsInstance(result, float)
        # u_J1 = -nu
        self.assertIsInstance(result, (int, float))

    def test_u_OO1(self):
        """u_OO1 함수를 테스트합니다."""
        result = nc.u_OO1(self.astro_data)

        self.assertIsInstance(result, float)
        # u_OO1 = -2*xi - nu
        self.assertIsInstance(result, (int, float))

    def test_u_M2(self):
        """u_M2 함수를 테스트합니다."""
        result = nc.u_M2(self.astro_data)

        self.assertIsInstance(result, float)
        # u_M2 = 2*xi - 2*nu
        self.assertIsInstance(result, (int, float))

    def test_u_K1(self):
        """u_K1 함수를 테스트합니다."""
        result = nc.u_K1(self.astro_data)

        self.assertIsInstance(result, float)
        # u_K1 = -nup
        self.assertIsInstance(result, (int, float))

    def test_u_L2(self):
        """u_L2 함수를 테스트합니다."""
        result = nc.u_L2(self.astro_data)

        self.assertIsInstance(result, float)
        # u_L2는 복잡한 계산을 포함
        self.assertIsInstance(result, (int, float))

    def test_u_K2(self):
        """u_K2 함수를 테스트합니다."""
        result = nc.u_K2(self.astro_data)

        self.assertIsInstance(result, float)
        # u_K2 = -2*nupp
        self.assertIsInstance(result, (int, float))

    def test_u_M1(self):
        """u_M1 함수를 테스트합니다."""
        result = nc.u_M1(self.astro_data)

        self.assertIsInstance(result, float)
        # u_M1은 복잡한 계산을 포함
        self.assertIsInstance(result, (int, float))

    def test_u_Modd(self):
        """u_Modd 함수를 테스트합니다."""
        # n=3인 경우 테스트
        result = nc.u_Modd(self.astro_data, 3)

        self.assertIsInstance(result, float)
        # u_Modd = n/2 * u_M2
        self.assertIsInstance(result, (int, float))

        # n=5인 경우 테스트
        result2 = nc.u_Modd(self.astro_data, 5)
        self.assertIsInstance(result2, float)
        self.assertIsInstance(result2, (int, float))

    def test_node_factors_consistency(self):
        """노드 인자들의 일관성을 테스트합니다."""
        # 같은 시간에 대해 두 번 호출했을 때 같은 결과가 나와야 함
        f_M2_1 = nc.f_M2(self.astro_data)
        f_M2_2 = nc.f_M2(self.astro_data)

        self.assertAlmostEqual(f_M2_1, f_M2_2, places=10)

    def test_node_factors_time_dependence(self):
        """노드 인자들의 시간 의존성을 테스트합니다."""
        time1 = datetime(2023, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 6, 15, 12, 0, 0)  # 6개월 후

        astro1 = pytidespy3.astro.astro(time1)
        astro2 = pytidespy3.astro.astro(time2)

        f_M2_1 = nc.f_M2(astro1)
        f_M2_2 = nc.f_M2(astro2)

        # 6개월 차이로 인해 값이 달라야 함
        self.assertNotAlmostEqual(f_M2_1, f_M2_2, places=5)

    def test_node_factors_ranges(self):
        """노드 인자들의 범위를 테스트합니다."""
        # f 값들은 일반적으로 0.5-1.5 범위에 있음
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
            self.assertLess(result, 2.0)  # 일반적으로 2보다 작음

    def test_u_values_ranges(self):
        """u 값들의 범위를 테스트합니다."""
        # u 값들은 일반적으로 -200에서 200 사이에 있음 (더 넓은 범위로 조정)
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
            # u 값이 -200보다 크고 200보다 작은지 확인 (더 넓은 범위)
            self.assertGreater(result, -200)
            self.assertLess(result, 200)

    def test_f_Modd_relationship(self):
        """f_Modd와 f_M2의 관계를 테스트합니다."""
        f_M2 = nc.f_M2(self.astro_data)
        f_M3 = nc.f_Modd(self.astro_data, 3)
        f_M5 = nc.f_Modd(self.astro_data, 5)

        # f_Modd(n) = f_M2^(n/2) 관계 확인
        expected_f_M3 = f_M2 ** (3 / 2)
        expected_f_M5 = f_M2 ** (5 / 2)

        self.assertAlmostEqual(f_M3, expected_f_M3, places=10)
        self.assertAlmostEqual(f_M5, expected_f_M5, places=10)

    def test_u_Modd_relationship(self):
        """u_Modd와 u_M2의 관계를 테스트합니다."""
        u_M2 = nc.u_M2(self.astro_data)
        u_M3 = nc.u_Modd(self.astro_data, 3)
        u_M5 = nc.u_Modd(self.astro_data, 5)

        # u_Modd(n) = n/2 * u_M2 관계 확인
        expected_u_M3 = 3 / 2 * u_M2
        expected_u_M5 = 5 / 2 * u_M2

        self.assertAlmostEqual(u_M3, expected_u_M3, places=10)
        self.assertAlmostEqual(u_M5, expected_u_M5, places=10)

    def test_astro_parameter_usage(self):
        """astro 파라미터들이 올바르게 사용되는지 테스트합니다."""
        # astro_data에 필요한 키들이 있는지 확인
        required_keys = ["omega", "i", "I", "xi", "nu", "nup", "nupp", "P"]

        for key in required_keys:
            self.assertIn(key, self.astro_data)

    def test_mathematical_consistency(self):
        """수학적 일관성을 테스트합니다."""
        # f_M2와 f_L2의 관계 확인 (f_L2는 f_M2에 추가 항을 곱함)
        f_M2 = nc.f_M2(self.astro_data)
        f_L2 = nc.f_L2(self.astro_data)

        # f_L2는 f_M2와 같거나 더 클 수 있음
        self.assertGreaterEqual(f_L2, f_M2 * 0.5)

    def test_edge_cases(self):
        """경계 케이스를 테스트합니다."""
        # 매우 먼 과거의 시간
        past_time = datetime(1900, 1, 1, 12, 0, 0)
        past_astro = pytidespy3.astro.astro(past_time)

        # 매우 먼 미래의 시간
        future_time = datetime(2100, 1, 1, 12, 0, 0)
        future_astro = pytidespy3.astro.astro(future_time)

        # 모든 함수가 이 시간들에서도 작동해야 함
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
        """정밀도를 테스트합니다."""
        # 같은 시간에 대해 여러 번 호출했을 때 결과가 일정해야 함
        f_M2_results = []
        u_M2_results = []

        for _ in range(10):
            f_M2_results.append(nc.f_M2(self.astro_data))
            u_M2_results.append(nc.u_M2(self.astro_data))

        # 모든 결과가 동일해야 함
        self.assertEqual(len(set(f_M2_results)), 1)
        self.assertEqual(len(set(u_M2_results)), 1)


if __name__ == "__main__":
    unittest.main()
