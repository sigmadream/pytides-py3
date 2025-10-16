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

        self.assertAlmostEqual(predicted_heights[0], predicted_heights[-1], delta=1.0)

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
            -0.454621527604,
            0.730974869360,
            -0.396309113917,
            -0.098231337180,
            0.006765310351,
            0.307388491194,
            -0.058753908925,
            -0.477433853629,
            0.481880495855,
        ])

        heights = self.tide_model.at(sample_times)
        np.testing.assert_allclose(heights, expected_heights, atol=1e-9)

        expected_extrema = [
            (datetime(2023, 1, 1, 3, 44, 1, 302107), 1.731447087847, "H"),
            (datetime(2023, 1, 1, 9, 43, 14, 339142), -1.320245792967, "L"),
            (datetime(2023, 1, 1, 15, 18, 5, 761490), 1.159819029553, "H"),
            (datetime(2023, 1, 1, 21, 3, 14, 812091), -1.598488864063, "L"),
            (datetime(2023, 1, 2, 3, 10, 54, 840221), 1.778754513583, "H"),
            (datetime(2023, 1, 2, 9, 11, 25, 125081), -1.369550218609, "L"),
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


if __name__ == "__main__":
    unittest.main()
