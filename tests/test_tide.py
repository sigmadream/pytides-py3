import unittest
import numpy as np
from datetime import datetime, timedelta
import pytidespy3.tide as tide
import pytidespy3.constituent as constituent
import pytidespy3.astro as astro


class TestTide(unittest.TestCase):
    """Test all methods of the Tide class."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)

        self.constituents = [constituent._M2, constituent._S2, constituent._K1]
        self.amplitudes = [1.0, 0.5, 0.3]
        self.phases = [0.0, 90.0, 180.0]

        self.tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=self.phases,
        )

    def test_init_with_constituents_amplitudes_phases(self):
        """Test initialization with constituents, amplitudes, and phases."""
        tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=self.phases,
        )

        self.assertIsInstance(tide_model.model, np.ndarray)
        self.assertEqual(len(tide_model.model), 3)
        self.assertEqual(tide_model.model.dtype, tide.Tide.dtype)

    def test_init_with_model(self):
        """Test initialization with model."""
        model = np.zeros(3, dtype=tide.Tide.dtype)
        model["constituent"] = self.constituents
        model["amplitude"] = self.amplitudes
        model["phase"] = self.phases

        tide_model = tide.Tide(model=model)

        self.assertIsInstance(tide_model.model, np.ndarray)
        self.assertEqual(len(tide_model.model), 3)

    def test_init_with_radians(self):
        """Test initialization with radians=True."""
        phases_radians = [0.0, np.pi / 2, np.pi]

        tide_model = tide.Tide(
            constituents=self.constituents,
            amplitudes=self.amplitudes,
            phases=phases_radians,
            radians=True,
        )

        expected_phases = [0.0, 90.0, 180.0]
        np.testing.assert_array_almost_equal(
            tide_model.model["phase"], expected_phases, decimal=5
        )

    def test_init_validation(self):
        """Test initialization validation."""
        with self.assertRaises(ValueError):
            tide.Tide(
                constituents=self.constituents,
                amplitudes=[1.0, 0.5],
                phases=self.phases,
            )

        with self.assertRaises(ValueError):
            tide.Tide()

        wrong_model = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            tide.Tide(model=wrong_model)

    def test_prepare(self):
        """Test prepare method."""
        t0 = self.test_time
        t = [self.test_time + timedelta(hours=i) for i in range(5)]

        speed, u, f, V0 = self.tide_model.prepare(t0, t)

        self.assertIsInstance(speed, np.ndarray)
        self.assertIsInstance(u, list)
        self.assertIsInstance(f, list)
        self.assertIsInstance(V0, np.ndarray)

        self.assertEqual(speed.shape[0], 3)
        self.assertEqual(len(u), 5)
        self.assertEqual(len(f), 5)
        self.assertEqual(V0.shape[0], 3)

    def test_prepare_radians_degrees(self):
        """Test prepare method with radians parameter."""
        t0 = self.test_time
        t = [self.test_time + timedelta(hours=i) for i in range(3)]

        speed_rad, u_rad, f_rad, V0_rad = self.tide_model.prepare(t0, t, radians=True)
        speed_deg, u_deg, f_deg, V0_deg = self.tide_model.prepare(t0, t, radians=False)

        self.assertLess(np.mean(speed_rad), np.mean(speed_deg))

    def test_at(self):
        """Test at method."""
        times = [self.test_time + timedelta(hours=i) for i in range(24)]

        heights = self.tide_model.at(times)

        self.assertIsInstance(heights, np.ndarray)
        self.assertEqual(len(heights), 24)

        self.assertTrue(np.all(np.isreal(heights)))

    def test_at_single_time(self):
        """Test at method with single time."""
        time = self.test_time

        height = self.tide_model.at([time])

        self.assertIsInstance(height, np.ndarray)
        self.assertEqual(len(height), 1)
        self.assertTrue(np.isreal(height[0]))

    def test_highs(self):
        """Test highs method."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        high_tides = list(self.tide_model.highs(t0, t1))

        self.assertIsInstance(high_tides, list)

        for high_tide in high_tides:
            self.assertEqual(len(high_tide), 3)
            self.assertIsInstance(high_tide[0], datetime)
            self.assertIsInstance(high_tide[1], (int, float))
            self.assertEqual(high_tide[2], "H")

    def test_lows(self):
        """Test lows method."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        low_tides = list(self.tide_model.lows(t0, t1))

        self.assertIsInstance(low_tides, list)

        for low_tide in low_tides:
            self.assertEqual(len(low_tide), 3)
            self.assertIsInstance(low_tide[0], datetime)
            self.assertIsInstance(low_tide[1], (int, float))
            self.assertEqual(low_tide[2], "L")

    def test_form_number(self):
        """Test form_number method."""
        form = self.tide_model.form_number()

        self.assertIsInstance(form, float)
        self.assertGreaterEqual(form, 0)

    def test_classify(self):
        """Test classify method."""
        classification = self.tide_model.classify()

        self.assertIsInstance(classification, str)
        self.assertIn(
            classification,
            ["semidiurnal", "mixed (semidiurnal)", "mixed (diurnal)", "diurnal"],
        )

    def test_extrema(self):
        """Test extrema method."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=3)

        extrema = list(self.tide_model.extrema(t0, t1))

        self.assertIsInstance(extrema, list)

        for extremum in extrema:
            self.assertEqual(len(extremum), 3)
            self.assertIsInstance(extremum[0], datetime)
            self.assertIsInstance(extremum[1], (int, float))
            self.assertIn(extremum[2], ["H", "L"])

    def test_extrema_infinite(self):
        """Test infinite extrema generator."""
        t0 = self.test_time

        extrema = []
        for i, extremum in enumerate(self.tide_model.extrema(t0)):
            if i >= 10:
                break
            extrema.append(extremum)

        self.assertEqual(len(extrema), 10)

    def test_hours(self):
        """Test _hours static method."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(hours=2)
        t2 = self.test_time + timedelta(hours=5)

        hours_single = tide.Tide._hours(t0, t1)
        self.assertIsInstance(hours_single, float)
        self.assertAlmostEqual(hours_single, 2.0, places=5)

        times = [t1, t2]
        hours_multiple = tide.Tide._hours(t0, times)
        self.assertIsInstance(hours_multiple, np.ndarray)
        self.assertEqual(len(hours_multiple), 2)
        np.testing.assert_array_almost_equal(hours_multiple, [2.0, 5.0], decimal=5)

    def test_partition(self):
        """Test _partition static method."""
        hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        partition = 3.0

        partitions = tide.Tide._partition(hours, partition)

        self.assertIsInstance(partitions, list)
        self.assertGreater(len(partitions), 0)

        for part in partitions:
            self.assertTrue(np.all(np.diff(part) >= 0))

    def test_times(self):
        """Test _times static method."""
        t0 = self.test_time
        hours = [0, 1, 2, 3]

        times = tide.Tide._times(t0, hours)

        self.assertIsInstance(times, list)
        self.assertEqual(len(times), 4)

        for i, time in enumerate(times):
            expected_time = t0 + timedelta(hours=i)
            self.assertEqual(time, expected_time)

    def test_tidal_series(self):
        """Test _tidal_series static method."""
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
        """Test normalize method."""
        original_model = self.tide_model.model.copy()

        self.tide_model.normalize()

        np.testing.assert_array_equal(self.tide_model.model, original_model)

    def test_decompose_class_method(self):
        """Test decompose class method."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(24)]

        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        decomposed_model = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2]
        )

        self.assertIsInstance(decomposed_model, tide.Tide)
        self.assertEqual(len(decomposed_model.model), 1)

    def test_decompose_with_initial_guess(self):
        """Test decompose with initial guess."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(24)]

        simple_model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )
        heights = simple_model.at(times)

        initial = np.zeros(2, dtype=tide.Tide.dtype)
        initial["constituent"] = [constituent._M2, constituent._S2]
        initial["amplitude"] = [0.8, 0.4]
        initial["phase"] = [10.0, 80.0]

        decomposed_model = tide.Tide.decompose(
            heights=heights,
            t=times,
            constituents=[constituent._M2, constituent._S2],
            initial=initial,
            n_period=0,
        )

        self.assertIsInstance(decomposed_model, tide.Tide)
        self.assertEqual(len(decomposed_model.model), 3)

    def test_decompose_with_callback(self):
        """Test decompose with callback."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(12)]

        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        callback_called = False

        def callback(residual):
            nonlocal callback_called
            callback_called = True
            self.assertIsInstance(residual, np.ndarray)

        decomposed_model = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2], callback=callback
        )

        self.assertTrue(callback_called)
        self.assertIsInstance(decomposed_model, tide.Tide)

    def test_decompose_full_output(self):
        """Test decompose with full_output=True."""
        t0 = self.test_time
        times = [t0 + timedelta(hours=i) for i in range(12)]

        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )
        heights = simple_model.at(times)

        result = tide.Tide.decompose(
            heights=heights, t=times, constituents=[constituent._M2], full_output=True
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tide.Tide)
        self.assertIsInstance(result[1], dict)

    def test_model_properties(self):
        """Test model properties."""
        self.assertEqual(self.tide_model.model.dtype, tide.Tide.dtype)

        expected_fields = ["constituent", "amplitude", "phase"]
        for field in expected_fields:
            self.assertIn(field, self.tide_model.model.dtype.names)

    def test_tide_prediction_accuracy(self):
        """Test tide prediction accuracy."""
        simple_model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )

        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        predicted_heights = simple_model.at(times)

        self.assertAlmostEqual(predicted_heights[0], predicted_heights[-1], delta=0.65)

    def test_tide_classification_consistency(self):
        """Test tide classification consistency."""
        classification1 = self.tide_model.classify()
        classification2 = self.tide_model.classify()

        self.assertEqual(classification1, classification2)

    def test_prediction_matches_golden_data(self):
        """Compare tidal predictions and extrema against golden data."""
        base_time = datetime(2023, 1, 1, 0, 0, 0)
        sample_hours = list(range(0, 49, 6))
        sample_times = [base_time + timedelta(hours=h) for h in sample_hours]
        expected_heights = np.array([
            -1.291415354521,
            0.905184475026,
            -0.598441116644,
            0.928739761142,
            -1.164352409967,
            0.742267114021,
            -0.390429940804,
            0.679387268523,
            -0.887571564512,
        ])

        heights = self.tide_model.at(sample_times)
        np.testing.assert_allclose(heights, expected_heights, atol=1e-9)

        expected_extrema = [
            (datetime(2023, 1, 1, 5, 45, 34, 574371), 0.915911024146, "H"),
            (datetime(2023, 1, 1, 11, 31, 16, 666570), -0.630804865364, "L"),
            (datetime(2023, 1, 1, 17, 25, 24, 332669), 0.973372473364, "H"),
            (datetime(2023, 1, 2, 0, 2, 4, 540310), -1.163829876367, "L"),
            (datetime(2023, 1, 2, 6, 39, 47, 732694), 0.775521231255, "H"),
            (datetime(2023, 1, 2, 12, 28, 10, 483534), -0.400454250753, "L"),
        ]

        extrema = list(self.tide_model.extrema(base_time, base_time + timedelta(days=2)))
        self.assertGreaterEqual(len(extrema), len(expected_extrema))

        for (time, height, kind), (expected_time, expected_height, expected_kind) in zip(extrema, expected_extrema):
            self.assertAlmostEqual((time - expected_time).total_seconds(), 0.0, places=6)
            self.assertAlmostEqual(height, expected_height, places=9)
            self.assertEqual(kind, expected_kind)

    def test_extrema_ordering(self):
        """Test extrema time ordering."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=7)

        extrema = list(self.tide_model.extrema(t0, t1))

        for i in range(len(extrema) - 1):
            self.assertLess(extrema[i][0], extrema[i + 1][0])

    def test_high_low_alternation(self):
        """Test high and low tide alternation."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(days=3)

        extrema = list(self.tide_model.extrema(t0, t1))

        for i in range(len(extrema) - 1):
            self.assertNotEqual(extrema[i][2], extrema[i + 1][2])

    def test_edge_cases(self):
        """Test edge cases."""
        t0 = self.test_time
        t1 = self.test_time + timedelta(minutes=30)

        extrema = list(self.tide_model.extrema(t0, t1))
        self.assertIsInstance(extrema, list)

        t1_long = self.test_time + timedelta(days=30)
        extrema_long = list(self.tide_model.extrema(t0, t1_long))
        self.assertGreater(len(extrema_long), 0)


class TestTideRobustness(unittest.TestCase):
    """결측/불규칙 간격/이상치 로버스트성 테스트."""

    def setUp(self):
        """합성 조석 데이터 생성 (M2 분조, 168시간)."""
        self.t0 = datetime(2023, 1, 1, 0, 0, 0)
        self.n_hours = 168
        self.times = [self.t0 + timedelta(hours=i) for i in range(self.n_hours)]
        self.source_model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 45.0],
        )
        self.clean_heights = self.source_model.at(self.times)

    def test_decompose_with_nan_heights(self):
        """heights에 NaN이 포함되어도 정상 분석되는지 확인."""
        heights = self.clean_heights.copy()
        # 10% 비율로 NaN 삽입
        rng = np.random.default_rng(42)
        nan_indices = rng.choice(len(heights), size=len(heights) // 10, replace=False)
        heights[nan_indices] = np.nan

        result = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
        )
        self.assertIsInstance(result, tide.Tide)
        # NaN이 제거된 후에도 분석 결과가 유효해야 함
        predicted = result.at(self.times)
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_decompose_with_inf_heights(self):
        """heights에 inf가 포함되어도 정상 분석되는지 확인."""
        heights = self.clean_heights.copy()
        heights[5] = np.inf
        heights[10] = -np.inf

        result = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
        )
        self.assertIsInstance(result, tide.Tide)
        predicted = result.at(self.times)
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_decompose_all_nan_raises(self):
        """모든 값이 NaN이면 ValueError가 발생해야 함."""
        heights = np.full(self.n_hours, np.nan)

        with self.assertRaises(ValueError):
            tide.Tide.decompose(
                heights=heights,
                t=self.times,
                constituents=[constituent._M2],
                n_period=0,
            )

    def test_decompose_with_weights(self):
        """weights 파라미터가 피팅 결과에 영향을 미치는지 확인."""
        heights = self.clean_heights.copy()
        # 이상치 삽입
        heights[50] += 5.0
        heights[100] += 5.0

        # 균등 가중치 (이상치 영향 큼)
        result_uniform = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
        )

        # 이상치에 낮은 가중치 적용
        weights = np.ones(self.n_hours)
        weights[50] = 0.01
        weights[100] = 0.01

        result_weighted = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
            weights=weights,
        )

        # 가중치를 적용한 결과가 원본에 더 가까워야 함
        pred_uniform = result_uniform.at(self.times)
        pred_weighted = result_weighted.at(self.times)
        rmse_uniform = np.sqrt(np.mean((self.clean_heights - pred_uniform) ** 2))
        rmse_weighted = np.sqrt(np.mean((self.clean_heights - pred_weighted) ** 2))
        self.assertLess(rmse_weighted, rmse_uniform)

    def test_decompose_with_robust_loss(self):
        """loss='huber' 사용 시 이상치에 강건한지 확인."""
        heights = self.clean_heights.copy()
        # 큰 이상치 삽입
        heights[30] += 10.0
        heights[80] += 10.0
        heights[130] -= 10.0

        result_linear = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
            loss='linear',
        )

        result_huber = tide.Tide.decompose(
            heights=heights,
            t=self.times,
            constituents=[constituent._M2, constituent._S2],
            n_period=0,
            loss='huber',
        )

        pred_linear = result_linear.at(self.times)
        pred_huber = result_huber.at(self.times)
        rmse_linear = np.sqrt(np.mean((self.clean_heights - pred_linear) ** 2))
        rmse_huber = np.sqrt(np.mean((self.clean_heights - pred_huber) ** 2))
        self.assertLess(rmse_huber, rmse_linear)

    def test_at_empty_array_raises(self):
        """빈 배열 입력 시 ValueError가 발생해야 함."""
        with self.assertRaises(ValueError):
            self.source_model.at([])


if __name__ == "__main__":
    unittest.main()
