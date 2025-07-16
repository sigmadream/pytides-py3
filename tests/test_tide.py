import unittest
import numpy as np
from datetime import datetime, timedelta
import pytidespy3.tide as tide
import pytidespy3.constituent as constituent
import pytidespy3.astro as astro


class TestTide(unittest.TestCase):
    """Tide 클래스의 모든 메서드를 테스트합니다."""

    def setUp(self):
        """테스트에 사용할 기본 설정을 합니다."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)

        # 테스트용 조화분조 모델 생성
        self.constituents = [constituent._M2, constituent._S2, constituent._K1]
        self.amplitudes = [1.0, 0.5, 0.3]
        self.phases = [0.0, 90.0, 180.0]

        self.tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=self.phases,
        )

    def test_init_with_constituents_amplitudes_phases(self):
        """constituents, amplitudes, phases로 초기화하는 것을 테스트합니다."""
        tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=self.phases,
        )

        self.assertIsInstance(tide_model.model, np.ndarray)
        self.assertEqual(len(tide_model.model), 3)
        self.assertEqual(tide_model.model.dtype, tide.Tide.dtype)

    def test_init_with_model(self):
        """model로 초기화하는 것을 테스트합니다."""
        model = np.zeros(3, dtype=tide.Tide.dtype)
        model["constituent"] = self.constituents
        model["amplitude"] = self.amplitudes
        model["phase"] = self.phases

        tide_model = tide.Tide(model=model)

        self.assertIsInstance(tide_model.model, np.ndarray)
        self.assertEqual(len(tide_model.model), 3)

    def test_init_with_radians(self):
        """radians=True로 초기화하는 것을 테스트합니다."""
        phases_radians = [0.0, np.pi / 2, np.pi]  # 라디안 단위

        tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=phases_radians,
            radians=True,
        )

        # 라디안이 도로 변환되었는지 확인
        expected_phases = [0.0, 90.0, 180.0]
        np.testing.assert_array_almost_equal(
            tide_model.model["phase"], expected_phases, decimal=5
        )

    def test_init_validation(self):
        """초기화 시 검증을 테스트합니다."""
        # 길이가 다른 경우
        with self.assertRaises(ValueError):
            tide.Tide(
                constituents=self.constituents,
                amplitudes=[1.0, 0.5],  # 길이가 다름
                phases=self.phases,
            )

        # model이 None이고 다른 파라미터도 None인 경우
        with self.assertRaises(ValueError):
            tide.Tide()

        # 잘못된 dtype의 model
        wrong_model = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            tide.Tide(model=wrong_model)

    def test_prepare(self):
        """prepare 메서드를 테스트합니다."""
        t0 = self.test_time
        t = [self.test_time + timedelta(hours=i) for i in range(5)]

        speed, u, f, V0 = self.tide_model.prepare(t0, t)

        self.assertIsInstance(speed, np.ndarray)
        self.assertIsInstance(u, list)
        self.assertIsInstance(f, list)
        self.assertIsInstance(V0, np.ndarray)

        # 차원 확인
        self.assertEqual(speed.shape[0], 3)  # 3개 조화분조
        self.assertEqual(len(u), 5)  # 5개 시간
        self.assertEqual(len(f), 5)
        self.assertEqual(V0.shape[0], 3)

    def test_prepare_radians_degrees(self):
        """prepare 메서드의 radians 파라미터를 테스트합니다."""
        t0 = self.test_time
        t = [self.test_time + timedelta(hours=i) for i in range(3)]

        # 라디안으로
        speed_rad, u_rad, f_rad, V0_rad = self.tide_model.prepare(t0, t, radians=True)
        # 도로
        speed_deg, u_deg, f_deg, V0_deg = self.tide_model.prepare(t0, t, radians=False)

        # 라디안 값이 도 값보다 작아야 함 (π/180 ≈ 0.017)
        self.assertLess(np.mean(speed_rad), np.mean(speed_deg))

    def test_at(self):
        """at 메서드를 테스트합니다."""
        times = [self.test_time + timedelta(hours=i) for i in range(24)]

        heights = self.tide_model.at(times)

        self.assertIsInstance(heights, np.ndarray)
        self.assertEqual(len(heights), 24)

        # 조석 높이는 실수여야 함
        self.assertTrue(np.all(np.isreal(heights)))

    def test_at_single_time(self):
        """단일 시간에 대한 at 메서드를 테스트합니다."""
        time = self.test_time

        height = self.tide_model.at([time])

        self.assertIsInstance(height, np.ndarray)
        self.assertEqual(len(height), 1)
        self.assertTrue(np.isreal(height[0]))

    def test_highs(self):
        """highs 메서드를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        high_tides = list(self.tide_model.highs(t0, t1))

        self.assertIsInstance(high_tides, list)

        # 각 고조는 (시간, 높이, 'H') 형태여야 함
        for high_tide in high_tides:
            self.assertEqual(len(high_tide), 3)
            self.assertIsInstance(high_tide[0], datetime)
            self.assertIsInstance(high_tide[1], (int, float))
            self.assertEqual(high_tide[2], "H")

    def test_lows(self):
        """lows 메서드를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        low_tides = list(self.tide_model.lows(t0, t1))

        self.assertIsInstance(low_tides, list)

        # 각 저조는 (시간, 높이, 'L') 형태여야 함
        for low_tide in low_tides:
            self.assertEqual(len(low_tide), 3)
            self.assertIsInstance(low_tide[0], datetime)
            self.assertIsInstance(low_tide[1], (int, float))
            self.assertEqual(low_tide[2], "L")

    def test_form_number(self):
        """form_number 메서드를 테스트합니다."""
        form = self.tide_model.form_number()

        self.assertIsInstance(form, float)
        self.assertGreaterEqual(form, 0)

    def test_classify(self):
        """classify 메서드를 테스트합니다."""
        classification = self.tide_model.classify()

        self.assertIsInstance(classification, str)
        self.assertIn(
            classification,
            ["semidiurnal", "mixed (semidiurnal)", "mixed (diurnal)", "diurnal"],
        )

    def test_extrema(self):
        """extrema 메서드를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=3)

        extrema = list(self.tide_model.extrema(t0, t1))

        self.assertIsInstance(extrema, list)

        # 각 극값은 (시간, 높이, 'H' 또는 'L') 형태여야 함
        for extremum in extrema:
            self.assertEqual(len(extremum), 3)
            self.assertIsInstance(extremum[0], datetime)
            self.assertIsInstance(extremum[1], (int, float))
            self.assertIn(extremum[2], ["H", "L"])

    def test_extrema_infinite(self):
        """무한 extrema 생성기를 테스트합니다."""
        t0 = self.test_time

        # 처음 10개의 극값만 가져오기
        extrema = []
        for i, extremum in enumerate(self.tide_model.extrema(t0)):
            if i >= 10:
                break
            extrema.append(extremum)

        self.assertEqual(len(extrema), 10)

    def test_hours(self):
        """_hours 정적 메서드를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(hours=2)
        t2 = self.test_time + timedelta(hours=5)

        # 단일 시간
        hours_single = tide.Tide._hours(t0, t1)
        self.assertIsInstance(hours_single, float)
        self.assertAlmostEqual(hours_single, 2.0, places=5)

        # 여러 시간
        times = [t1, t2]
        hours_multiple = tide.Tide._hours(t0, times)
        self.assertIsInstance(hours_multiple, np.ndarray)
        self.assertEqual(len(hours_multiple), 2)
        np.testing.assert_array_almost_equal(hours_multiple, [2.0, 5.0], decimal=5)

    def test_partition(self):
        """_partition 정적 메서드를 테스트합니다."""
        hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        partition = 3.0

        partitions = tide.Tide._partition(hours, partition)

        self.assertIsInstance(partitions, list)
        self.assertGreater(len(partitions), 0)

        # 각 파티션의 시간들이 정렬되어 있어야 함
        for part in partitions:
            self.assertTrue(np.all(np.diff(part) >= 0))

    def test_times(self):
        """_times 정적 메서드를 테스트합니다."""
        t0 = self.test_time
        hours = [0, 1, 2, 3]

        times = tide.Tide._times(t0, hours)

        self.assertIsInstance(times, list)
        self.assertEqual(len(times), 4)

        # 각 시간이 올바르게 계산되었는지 확인
        for i, time in enumerate(times):
            expected_time = t0 + timedelta(hours=i)
            self.assertEqual(time, expected_time)

    def test_tidal_series(self):
        """_tidal_series 정적 메서드를 테스트합니다."""
        t = np.array([0, 1, 2, 3, 4])
        amplitude = np.array([[1.0], [0.5]])
        phase = np.array([[0.0], [90.0]])
        speed = np.array([[15.0], [30.0]])
        u = np.array([[0.0], [0.0]])
        f = np.array([[1.0], [1.0]])
        V0 = np.array([[0.0], [0.0]])

        series = tide.Tide._tidal_series(t, amplitude, phase, speed, u, f, V0)

        self.assertIsInstance(series, np.ndarray)
        self.assertEqual(len(series), 5)
        self.assertTrue(np.all(np.isreal(series)))

    def test_normalize(self):
        """normalize 메서드를 테스트합니다."""
        # 정규화 전후의 모델이 같은지 확인
        original_model = self.tide_model.model.copy()

        self.tide_model.normalize()

        # 정규화는 모델을 변경하지 않아야 함 (이미 정규화되어 있음)
        np.testing.assert_array_equal(self.tide_model.model, original_model)

    def test_decompose_class_method(self):
        """decompose 클래스 메서드를 테스트합니다."""
        # 간단한 조석 데이터 생성
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(24)]

        # 간단한 조석 모델로 높이 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        # 분해
        decomposed_model = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2]
        )

        self.assertIsInstance(decomposed_model, tide.Tide)
        self.assertEqual(len(decomposed_model.model), 1)

    def test_decompose_with_initial_guess(self):
        """초기 추정치가 있는 decompose를 테스트합니다."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(24)]

        # 간단한 조석 모델로 높이 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )
        heights = simple_model.at(times)

        # 초기 추정치
        initial = np.zeros(2, dtype=tide.Tide.dtype)
        initial["constituent"] = [constituent._M2, constituent._S2]
        initial["amplitude"] = [0.8, 0.4]
        initial["phase"] = [10.0, 80.0]

        # 분해
        decomposed_model = tide.Tide.decompose(
            heights=heights,
            t=times,
            constituents=[constituent._M2, constituent._S2],
            initial=initial,
        )

        self.assertIsInstance(decomposed_model, tide.Tide)
        self.assertEqual(len(decomposed_model.model), 2)

    def test_decompose_with_callback(self):
        """콜백이 있는 decompose를 테스트합니다."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(12)]

        # 간단한 조석 모델로 높이 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        callback_called = False

        def callback(iteration, residual):
            nonlocal callback_called
            callback_called = True
            self.assertIsInstance(iteration, int)
            self.assertIsInstance(residual, float)

        # 분해
        decomposed_model = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2], callback=callback
        )

        self.assertTrue(callback_called)
        self.assertIsInstance(decomposed_model, tide.Tide)

    def test_decompose_full_output(self):
        """full_output=True인 decompose를 테스트합니다."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(12)]

        # 간단한 조석 모델로 높이 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        # 분해
        result = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2], full_output=True
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tide.Tide)
        self.assertIsInstance(result[1], dict)

    def test_model_properties(self):
        """모델의 속성들을 테스트합니다."""
        # 모델의 dtype 확인
        self.assertEqual(self.tide_model.model.dtype, tide.Tide.dtype)

        # 모델의 필드 확인
        expected_fields = ["constituent", "amplitude", "phase"]
        for field in expected_fields:
            self.assertIn(field, self.tide_model.model.dtype.names)

    def test_tide_prediction_accuracy(self):
        """조석 예측의 정확성을 테스트합니다."""
        # 간단한 조석 모델 생성
        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )

        # 24시간 동안의 조석 예측
        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        predicted_heights = simple_model.at(times)

        # M2 조화분조는 약 12.42시간 주기
        # 24시간 후에는 거의 같은 높이여야 함
        self.assertAlmostEqual(predicted_heights[0], predicted_heights[-1], places=2)

    def test_tide_classification_consistency(self):
        """조석 분류의 일관성을 테스트합니다."""
        # 같은 모델에 대해 분류가 일관되어야 함
        classification1 = self.tide_model.classify()
        classification2 = self.tide_model.classify()

        self.assertEqual(classification1, classification2)

    def test_extrema_ordering(self):
        """극값들의 시간 순서를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        extrema = list(self.tide_model.extrema(t0, t1))

        # 극값들이 시간 순서대로 정렬되어 있어야 함
        for i in range(len(extrema) - 1):
            self.assertLess(extrema[i][0], extrema[i + 1][0])

    def test_high_low_alternation(self):
        """고조와 저조의 교대를 테스트합니다."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=3)

        extrema = list(self.tide_model.extrema(t0, t1))

        # 고조와 저조가 교대로 나타나야 함
        for i in range(len(extrema) - 1):
            self.assertNotEqual(extrema[i][2], extrema[i + 1][2])

    def test_edge_cases(self):
        """경계 케이스를 테스트합니다."""
        # 매우 짧은 시간 간격
        t0 = self.test_time
        t1 = self.test_time + timedelta(minutes=30)

        extrema = list(self.tide_model.extrema(t0, t1))
        # 매우 짧은 시간에는 극값이 없을 수 있음
        self.assertIsInstance(extrema, list)

        # 매우 긴 시간 간격
        t1_long = self.test_time + timedelta(days=30)
        extrema_long = list(self.tide_model.extrema(t0, t1_long))
        self.assertGreater(len(extrema_long), 0)


if __name__ == "__main__":
    unittest.main()
