import unittest
import numpy as np
from datetime import datetime, timedelta
from pytidespy3.astro import *


class TestAstro(unittest.TestCase):
    """astro 모듈의 모든 함수를 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 시간을 설정합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)  # 2023년 1월 1일 12시
        self.test_time2 = datetime(2023, 6, 15, 6, 30, 0)  # 2023년 6월 15일 6시 30분

    def test_s2d_basic(self):
        """s2d 함수의 기본 기능을 테스트합니다."""
        # 기본 각도 변환
        self.assertAlmostEqual(s2d(30, 15, 45), 30.2625, places=6)
        self.assertAlmostEqual(s2d(0, 30, 0), 0.5, places=6)
        self.assertAlmostEqual(s2d(45), 45.0, places=6)

    def test_s2d_edge_cases(self):
        """s2d 함수의 경계 케이스를 테스트합니다."""
        # 0도 테스트
        self.assertEqual(s2d(0, 0, 0), 0.0)
        # 음수 각도 테스트
        self.assertAlmostEqual(s2d(-30, 15, 45), -29.7375, places=6)

    def test_polynomial_basic(self):
        """polynomial 함수의 기본 기능을 테스트합니다."""
        coeffs = [1, 2, 3]  # 1 + 2x + 3x^2
        self.assertEqual(polynomial(coeffs, 0), 1)
        self.assertEqual(polynomial(coeffs, 1), 6)  # 1 + 2 + 3
        self.assertEqual(polynomial(coeffs, 2), 17)  # 1 + 4 + 12

    def test_polynomial_empty(self):
        """polynomial 함수의 빈 계수 리스트를 테스트합니다."""
        self.assertEqual(polynomial([], 5), 0)

    def test_d_polynomial_basic(self):
        """d_polynomial 함수의 기본 기능을 테스트합니다."""
        coeffs = [1, 2, 3]  # 1 + 2x + 3x^2, 도함수: 2 + 6x
        self.assertEqual(d_polynomial(coeffs, 0), 2)
        self.assertEqual(d_polynomial(coeffs, 1), 8)  # 2 + 6
        self.assertEqual(d_polynomial(coeffs, 2), 14)  # 2 + 12

    def test_d_polynomial_empty(self):
        """d_polynomial 함수의 빈 계수 리스트를 테스트합니다."""
        self.assertEqual(d_polynomial([], 5), 0)

    def test_T_basic(self):
        """T 함수의 기본 기능을 테스트합니다."""
        # Julian Centuries 계산 테스트
        result = T(self.test_time)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_T_different_times(self):
        """T 함수의 다른 시간들에 대한 테스트입니다."""
        time1 = datetime(2000, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 1, 1, 12, 0, 0)

        T1 = T(time1)
        T2 = T(time2)

        self.assertGreater(T2, T1)  # 2023년이 2000년보다 더 큰 T값을 가져야 함

    def test_JD_basic(self):
        """JD 함수의 기본 기능을 테스트합니다."""
        # Julian Day 계산 테스트
        result = JD(self.test_time)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 2400000)  # 2023년의 Julian Day는 2400000보다 큼

    def test_JD_edge_cases(self):
        """JD 함수의 경계 케이스를 테스트합니다."""
        # 2월 이전 날짜 테스트
        jan_date = datetime(2023, 1, 15, 12, 0, 0)
        feb_date = datetime(2023, 2, 15, 12, 0, 0)

        JD_jan = JD(jan_date)
        JD_feb = JD(feb_date)

        self.assertGreater(JD_feb, JD_jan)

    def test_astro_basic(self):
        """astro 함수의 기본 기능을 테스트합니다."""
        result = astro(self.test_time)

        # 결과가 딕셔너리인지 확인
        self.assertIsInstance(result, dict)

        # 필수 키들이 있는지 확인
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
        """astro 함수가 반환하는 각 파라미터의 구조를 테스트합니다."""
        result = astro(self.test_time)

        # 각 파라미터가 AstronomicalParameter 타입인지 확인
        for key, param in result.items():
            if key not in [
                "I",
                "xi",
                "nu",
                "nup",
                "nupp",
                "P",
            ]:  # 이들은 speed가 None일 수 있음
                self.assertIsInstance(param, AstronomicalParameter)
                self.assertIsInstance(param.value, (int, float))
                if param.speed is not None:
                    self.assertIsInstance(param.speed, (int, float))

    def test_astro_value_ranges(self):
        """astro 함수가 반환하는 값들의 범위를 테스트합니다."""
        result = astro(self.test_time)

        # 각도 값들이 0-360 범위에 있는지 확인
        for key, param in result.items():
            if hasattr(param, "value"):
                self.assertGreaterEqual(param.value, 0)
                self.assertLessEqual(param.value, 360)

    def test_astro_consistency(self):
        """astro 함수의 일관성을 테스트합니다."""
        # 같은 시간에 대해 두 번 호출했을 때 같은 결과가 나와야 함
        result1 = astro(self.test_time)
        result2 = astro(self.test_time)

        for key in result1:
            if hasattr(result1[key], "value"):
                self.assertAlmostEqual(
                    result1[key].value, result2[key].value, places=10
                )

    def test_astro_time_progression(self):
        """시간 진행에 따른 astro 함수의 변화를 테스트합니다."""
        time1 = datetime(2023, 1, 1, 12, 0, 0)
        time2 = datetime(2023, 1, 1, 13, 0, 0)  # 1시간 후

        result1 = astro(time1)
        result2 = astro(time2)

        # T+h-s는 시간에 따라 변화해야 함
        self.assertNotAlmostEqual(
            result1["T+h-s"].value, result2["T+h-s"].value, places=5
        )

    def test_astro_speed_values(self):
        """astro 함수가 반환하는 속도 값들을 테스트합니다."""
        result = astro(self.test_time)

        # 주요 천체들의 속도가 예상 범위에 있는지 확인
        # 태양의 속도 (h)는 약 0.041도/시간 (하루에 약 1도)
        self.assertAlmostEqual(result["h"].speed, 0.041, places=3)

        # 달의 속도 (s)는 약 0.549도/시간 (하루에 약 13.2도)
        self.assertAlmostEqual(result["s"].speed, 0.549, places=3)

    def test_astro_parameter_relationships(self):
        """astro 함수가 반환하는 파라미터들 간의 관계를 테스트합니다."""
        result = astro(self.test_time)

        # T+h-s = T + h - s 관계 확인
        T_plus_h_minus_s = result["T+h-s"].value
        T_value = (JD(self.test_time) - int(JD(self.test_time))) * 360.0
        h_value = result["h"].value
        s_value = result["s"].value

        expected = (T_value + h_value - s_value) % 360.0
        self.assertAlmostEqual(T_plus_h_minus_s, expected, places=5)

    def test_astro_different_years(self):
        """다른 연도에 대한 astro 함수의 동작을 테스트합니다."""
        time_2000 = datetime(2000, 1, 1, 12, 0, 0)
        time_2023 = datetime(2023, 1, 1, 12, 0, 0)

        result_2000 = astro(time_2000)
        result_2023 = astro(time_2023)

        # 23년 차이로 인해 일부 값들이 달라야 함
        self.assertNotAlmostEqual(
            result_2000["h"].value, result_2023["h"].value, places=5
        )

    def test_astro_microsecond_precision(self):
        """마이크로초 정밀도에 대한 astro 함수의 동작을 테스트합니다."""
        time1 = datetime(2023, 1, 1, 12, 0, 0, 0)
        time2 = datetime(2023, 1, 1, 12, 0, 0, 100000)  # 0.1초 차이

        result1 = astro(time1)
        result2 = astro(time2)

        # 매우 작은 차이이지만 값이 달라야 함
        self.assertNotEqual(result1["T+h-s"].value, result2["T+h-s"].value)


if __name__ == "__main__":
    unittest.main()
