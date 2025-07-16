import unittest
import numpy as np
from datetime import datetime, timedelta
import pytidespy3.tide as tide
import pytidespy3.constituent as constituent
import pytidespy3.astro
import pytidespy3.nodal_corrections as nc


class TestIntegration(unittest.TestCase):
    """모든 모듈이 함께 작동하는지 확인하는 통합 테스트입니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)

    def test_full_tide_analysis_workflow(self):
        """전체 조석 분석 워크플로우를 테스트합니다."""
        # 1. 천문학적 파라미터 계산
        astro_data = pytidespy3.astro.astro(self.test_time)
        self.assertIsInstance(astro_data, dict)

        # 2. 조화분조 정의
        constituents = [constituent._M2, constituent._S2, constituent._K1]
        self.assertEqual(len(constituents), 3)

        # 3. 조석 모델 생성
        amplitudes = [1.0, 0.5, 0.3]
        phases = [0.0, 90.0, 180.0]

        tide_model = tide.Tide(
            constituents=constituents, amplitudes=amplitudes, phases=phases
        )
        self.assertIsInstance(tide_model, tide.Tide)

        # 4. 조석 예측
        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        heights = tide_model.at(times)
        self.assertEqual(len(heights), 24)

        # 5. 고조/저조 예측
        high_tides = list(
            tide_model.highs(self.test_time, self.test_time + timedelta(days=7))
        )
        low_tides = list(
            tide_model.lows(self.test_time, self.test_time + timedelta(days=7))
        )

        self.assertGreater(len(high_tides), 0)
        self.assertGreater(len(low_tides), 0)

        # 6. 조석 분류
        classification = tide_model.classify()
        self.assertIsInstance(classification, str)

    def test_constituent_node_factors_integration(self):
        """조화분조와 노드 인자의 통합을 테스트합니다."""
        astro_data = pytidespy3.astro.astro(self.test_time)

        # 주요 조화분조들의 노드 인자 계산
        constituents = [
            constituent._M2,
            constituent._S2,
            constituent._K1,
            constituent._O1,
        ]

        for const in constituents:
            # V 값 계산
            V = const.V(astro_data)
            self.assertGreaterEqual(V, 0)
            self.assertLessEqual(V, 360)

            # 속도 계산
            speed = const.speed(astro_data)
            self.assertGreater(speed, 0)

            # 노드 인자 계산
            u = const.u(astro_data)
            f = const.f(astro_data)

            self.assertIsInstance(u, (int, float))
            self.assertGreater(f, 0)

    def test_tide_decomposition_workflow(self):
        """조석 분해 워크플로우를 테스트합니다."""
        # 1. 간단한 조석 데이터 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        # 2. 조석 데이터 생성
        times = [self.test_time + timedelta(hours=i) for i in range(48)]
        original_heights = simple_model.at(times)

        # 3. 조석 분해
        decomposed_model = tide.Tide.decompose(
            heights=original_heights,
            t=times,
            constituents=[constituent._M2, constituent._S2],
        )

        # 4. 분해된 모델로 재구성
        reconstructed_heights = decomposed_model.at(times)

        # 5. 원본과 재구성된 데이터 비교
        # 일정한 오차 범위 내에서 일치해야 함
        mse = np.mean((original_heights - reconstructed_heights) ** 2)
        self.assertLess(mse, 0.1)  # 오차가 0.1보다 작아야 함

    def test_astro_constituent_integration(self):
        """천문학적 파라미터와 조화분조의 통합을 테스트합니다."""
        # 다른 시간들에 대한 천문학적 파라미터
        times = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 6, 15, 12, 0, 0),
            datetime(2023, 12, 31, 12, 0, 0),
        ]

        constituents = [constituent._M2, constituent._S2, constituent._K1]

        for t in times:
            astro_data = pytidespy3.astro.astro(t)

            for const in constituents:
                # 각 시간에 대해 조화분조 계산이 가능해야 함
                V = const.V(astro_data)
                speed = const.speed(astro_data)
                u = const.u(astro_data)
                f = const.f(astro_data)

                # 모든 값이 유효해야 함
                self.assertIsInstance(V, (int, float))
                self.assertIsInstance(speed, (int, float))
                self.assertIsInstance(u, (int, float))
                self.assertIsInstance(f, (int, float))

                self.assertGreaterEqual(V, 0)
                self.assertLessEqual(V, 360)
                self.assertGreater(speed, 0)
                self.assertGreater(f, 0)

    def test_tide_prediction_consistency(self):
        """조석 예측의 일관성을 테스트합니다."""
        # 같은 모델로 다른 시간에 대해 예측
        model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        # 24시간 간격으로 예측
        base_time = self.test_time
        predictions = []

        for i in range(7):  # 7일간
            start_time = base_time + timedelta(days=i)
            times = [start_time + timedelta(hours=j) for j in range(24)]
            heights = model.at(times)
            predictions.append(heights)

        # M2 조화분조는 약 12.42시간 주기이므로
        # 24시간 후의 패턴이 반복되어야 함
        for i in range(6):
            # 각 날의 첫 번째와 마지막 높이 비교
            first_day_start = predictions[i][0]
            first_day_end = predictions[i][-1]
            second_day_start = predictions[i + 1][0]

            # 연속된 날의 시작과 끝이 비슷해야 함 (조화분조의 위상 차이로 인해 완전히 일치하지 않을 수 있음)
            self.assertAlmostEqual(first_day_end, second_day_start, delta=0.5)

    def test_extrema_prediction_consistency(self):
        """극값 예측의 일관성을 테스트합니다."""
        model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        # 7일간의 극값 예측
        start_time = self.test_time
        end_time = start_time + timedelta(days=7)

        extrema = list(model.extrema(start_time, end_time))

        # 극값들이 시간 순서대로 정렬되어 있어야 함
        for i in range(len(extrema) - 1):
            self.assertLess(extrema[i][0], extrema[i + 1][0])

        # 고조와 저조가 교대로 나타나야 함
        for i in range(len(extrema) - 1):
            self.assertNotEqual(extrema[i][2], extrema[i + 1][2])

        # M2 조화분조는 약 12.42시간 주기이므로
        # 하루에 약 2개의 고조와 2개의 저조가 있어야 함
        high_tides = [e for e in extrema if e[2] == "H"]
        low_tides = [e for e in extrema if e[2] == "L"]

        # 7일이므로 약 14개의 고조와 저조가 있어야 함
        self.assertGreaterEqual(len(high_tides), 10)
        self.assertGreaterEqual(len(low_tides), 10)

    def test_form_number_classification_consistency(self):
        """형수와 분류의 일관성을 테스트합니다."""
        # 다양한 조화분조 조합으로 테스트
        test_cases = [
            # 반일주조 (M2, S2만)
            ([constituent._M2, constituent._S2], [1.0, 0.5], [0.0, 90.0]),
            # 혼합조 (M2, S2, K1)
            (
                [constituent._M2, constituent._S2, constituent._K1],
                [1.0, 0.5, 0.3],
                [0.0, 90.0, 180.0],
            ),
            # 일주조 (K1, O1만)
            ([constituent._K1, constituent._O1], [1.0, 0.5], [0.0, 90.0]),
        ]

        for constituents, amplitudes, phases in test_cases:
            model = tide.Tide(
                constituents=constituents, amplitudes=amplitudes, phases=phases
            )

            form_number = model.form_number()
            classification = model.classify()

            # 형수가 양수여야 함
            self.assertGreaterEqual(form_number, 0)

            # 분류가 유효한 값이어야 함
            valid_classifications = [
                "semidiurnal",
                "mixed (semidiurnal)",
                "mixed (diurnal)",
                "diurnal",
            ]
            self.assertIn(classification, valid_classifications)

    def test_noaa_constituents_integration(self):
        """NOAA 조화분조들의 통합을 테스트합니다."""
        # NOAA 조화분조 리스트 사용
        noaa_constituents = constituent.noaa[:5]  # 처음 5개만 사용

        # 간단한 진폭과 위상
        amplitudes = [0.1] * len(noaa_constituents)
        phases = [0.0] * len(noaa_constituents)

        # 조석 모델 생성
        model = tide.Tide(
            constituents=noaa_constituents, amplitudes=amplitudes, phases=phases
        )

        # 조석 예측
        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        heights = model.at(times)

        self.assertEqual(len(heights), 24)
        self.assertTrue(np.all(np.isreal(heights)))

        # 고조/저조 예측
        high_tides = list(
            model.highs(self.test_time, self.test_time + timedelta(days=3))
        )
        low_tides = list(model.lows(self.test_time, self.test_time + timedelta(days=3)))

        self.assertGreater(len(high_tides), 0)
        self.assertGreater(len(low_tides), 0)

    def test_error_handling_integration(self):
        """통합된 오류 처리를 테스트합니다."""
        # 잘못된 조화분조 조합
        with self.assertRaises(ValueError):
            tide.Tide(
                constituents=[constituent._M2],
                amplitudes=[1.0, 0.5],  # 길이가 다름
                phases=[0.0],
            )

        # 잘못된 시간 형식
        model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )

        # 잘못된 시간 형식으로 예측 시도
        with self.assertRaises(Exception):
            model.at([1, 2, 3])  # datetime이 아닌 값들

    def test_performance_integration(self):
        """통합 성능을 테스트합니다."""
        # 큰 데이터셋으로 성능 테스트
        model = tide.Tide(
            constituents=constituent.noaa[:10],  # 10개 조화분조
            amplitudes=[0.1] * 10,
            phases=[0.0] * 10,
        )

        # 30일간의 조석 예측
        times = [self.test_time + timedelta(hours=i) for i in range(24 * 30)]

        # 예측 시간 측정
        import time

        start_time = time.time()
        heights = model.at(times)
        end_time = time.time()

        # 30일 예측이 합리적인 시간 내에 완료되어야 함
        self.assertLess(end_time - start_time, 10.0)  # 10초 이내
        self.assertEqual(len(heights), 24 * 30)

        # 극값 예측도 합리적인 시간 내에 완료되어야 함
        start_time = time.time()
        extrema = list(
            model.extrema(self.test_time, self.test_time + timedelta(days=30))
        )
        end_time = time.time()

        self.assertLess(end_time - start_time, 30.0)  # 30초 이내
        self.assertGreater(len(extrema), 0)


if __name__ == "__main__":
    unittest.main()
