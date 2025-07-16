import unittest
import numpy as np
from datetime import datetime, timedelta
import pytidespy3.tide as tide
import pytidespy3.constituent as constituent
import pytidespy3.astro
import pytidespy3.nodal_corrections as nc


class TestIntegration(unittest.TestCase):
    """Integration tests to verify all modules work together."""

    def setUp(self):
        """Set up basic test configuration."""
        self.test_time = datetime(2023, 1, 1, 12, 0, 0)

    def test_full_tide_analysis_workflow(self):
        """Test complete tidal analysis workflow."""
        astro_data = pytidespy3.astro.astro(self.test_time)
        self.assertIsInstance(astro_data, dict)

        constituents = [constituent._M2, constituent._S2, constituent._K1]
        self.assertEqual(len(constituents), 3)

        amplitudes = [1.0, 0.5, 0.3]
        phases = [0.0, 90.0, 180.0]

        tide_model = tide.Tide(
            constituents=constituents, amplitudes=amplitudes, phases=phases
        )
        self.assertIsInstance(tide_model, tide.Tide)

        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        heights = tide_model.at(times)
        self.assertEqual(len(heights), 24)

        high_tides = list(
            tide_model.highs(self.test_time, self.test_time + timedelta(days=7))
        )
        low_tides = list(
            tide_model.lows(self.test_time, self.test_time + timedelta(days=7))
        )

        self.assertGreater(len(high_tides), 0)
        self.assertGreater(len(low_tides), 0)

        classification = tide_model.classify()
        self.assertIsInstance(classification, str)

    def test_constituent_node_factors_integration(self):
        """Test integration of harmonic constituents and node factors."""
        astro_data = pytidespy3.astro.astro(self.test_time)

        constituents = [
            constituent._M2,
            constituent._S2,
            constituent._K1,
            constituent._O1,
        ]

        for const in constituents:
            V = const.V(astro_data)
            if isinstance(V, np.ndarray):
                V = float(V)
            self.assertGreaterEqual(V, 0)
            self.assertLessEqual(V, 360)

            speed = const.speed(astro_data)
            if isinstance(speed, np.ndarray):
                speed = float(speed)
            self.assertGreater(speed, 0)

            u = const.u(astro_data)
            f = const.f(astro_data)

            self.assertIsInstance(u, (int, float))
            if isinstance(f, np.ndarray):
                f = float(f)
            self.assertGreater(f, 0)

    def test_tide_decomposition_workflow(self):
        """Test tidal decomposition workflow."""
        simple_model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        times = [self.test_time + timedelta(hours=i) for i in range(48)]
        original_heights = simple_model.at(times)

        decomposed_result = tide.Tide.decompose(
            heights=original_heights,
            t=times,
            constituents=[constituent._M2, constituent._S2],
        )
        
        if isinstance(decomposed_result, tuple):
            decomposed_model = decomposed_result[0]
        else:
            decomposed_model = decomposed_result

        reconstructed_heights = decomposed_model.at(times)

        mse = np.mean((original_heights - reconstructed_heights) ** 2)
        self.assertLess(mse, 0.1)

    def test_astro_constituent_integration(self):
        """Test integration of astronomical parameters and harmonic constituents."""
        times = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 6, 15, 12, 0, 0),
            datetime(2023, 12, 31, 12, 0, 0),
        ]

        constituents = [constituent._M2, constituent._S2, constituent._K1]

        for t in times:
            astro_data = pytidespy3.astro.astro(t)

            for const in constituents:
                V = const.V(astro_data)
                speed = const.speed(astro_data)
                u = const.u(astro_data)
                f = const.f(astro_data)

                if isinstance(V, np.ndarray):
                    V = float(V)
                if isinstance(speed, np.ndarray):
                    speed = float(speed)
                if isinstance(f, np.ndarray):
                    f = float(f)

                self.assertIsInstance(V, (int, float))
                self.assertIsInstance(speed, (int, float))
                self.assertIsInstance(u, (int, float))
                self.assertIsInstance(f, (int, float))

                self.assertGreaterEqual(V, 0)
                self.assertLessEqual(V, 360)
                self.assertGreater(speed, 0)
                self.assertGreater(f, 0)

    def test_tide_prediction_consistency(self):
        """Test consistency of tide predictions."""
        model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        base_time = self.test_time
        predictions = []

        for i in range(7):
            start_time = base_time + timedelta(days=i)
            times = [start_time + timedelta(hours=j) for j in range(24)]
            heights = model.at(times)
            predictions.append(heights)

        for i in range(6):
            first_day_start = predictions[i][0]
            first_day_end = predictions[i][-1]
            second_day_start = predictions[i + 1][0]

            self.assertAlmostEqual(first_day_end, second_day_start, delta=1.5)

    def test_extrema_prediction_consistency(self):
        """Test consistency of extrema predictions."""
        model = tide.Tide(
            constituents=[constituent._M2, constituent._S2],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 90.0],
        )

        start_time = self.test_time
        end_time = start_time + timedelta(days=7)

        extrema = list(model.extrema(start_time, end_time))

        for i in range(len(extrema) - 1):
            self.assertLess(extrema[i][0], extrema[i + 1][0])

        for i in range(len(extrema) - 1):
            self.assertNotEqual(extrema[i][2], extrema[i + 1][2])

        high_tides = [e for e in extrema if e[2] == "H"]
        low_tides = [e for e in extrema if e[2] == "L"]

        self.assertGreaterEqual(len(high_tides), 10)
        self.assertGreaterEqual(len(low_tides), 10)

    def test_form_number_classification_consistency(self):
        """Test consistency of form number and classification."""
        test_cases = [
            ([constituent._M2, constituent._S2], [1.0, 0.5], [0.0, 90.0]),
            (
                [constituent._M2, constituent._S2, constituent._K1],
                [1.0, 0.5, 0.3],
                [0.0, 90.0, 180.0],
            ),
            ([constituent._K1, constituent._O1], [1.0, 0.5], [0.0, 90.0]),
        ]

        for constituents, amplitudes, phases in test_cases:
            model = tide.Tide(
                constituents=constituents, amplitudes=amplitudes, phases=phases
            )

            form_number = model.form_number()
            classification = model.classify()

            self.assertGreaterEqual(form_number, 0)

            valid_classifications = [
                "semidiurnal",
                "mixed (semidiurnal)",
                "mixed (diurnal)",
                "diurnal",
            ]
            self.assertIn(classification, valid_classifications)

    def test_noaa_constituents_integration(self):
        """Test integration with NOAA harmonic constituents."""
        noaa_constituents = constituent.noaa[:5]

        amplitudes = [0.1] * len(noaa_constituents)
        phases = [0.0] * len(noaa_constituents)

        model = tide.Tide(
            constituents=noaa_constituents, amplitudes=amplitudes, phases=phases
        )

        times = [self.test_time + timedelta(hours=i) for i in range(24)]
        heights = model.at(times)

        self.assertEqual(len(heights), 24)
        self.assertTrue(np.all(np.isreal(heights)))

        high_tides = list(
            model.highs(self.test_time, self.test_time + timedelta(days=3))
        )
        low_tides = list(model.lows(self.test_time, self.test_time + timedelta(days=3)))

        self.assertGreater(len(high_tides), 0)
        self.assertGreater(len(low_tides), 0)

    def test_error_handling_integration(self):
        """Test integrated error handling."""
        with self.assertRaises(ValueError):
            tide.Tide(
                constituents=[constituent._M2],
                amplitudes=[1.0, 0.5],
                phases=[0.0],
            )

        model = tide.Tide(
            constituents=[constituent._M2], amplitudes=[1.0], phases=[0.0]
        )

        with self.assertRaises(Exception):
            model.at([1, 2, 3])

    def test_performance_integration(self):
        """Test integration performance."""
        model = tide.Tide(
            constituents=constituent.noaa[:10],
            amplitudes=[0.1] * 10,
            phases=[0.0] * 10,
        )

        times = [self.test_time + timedelta(hours=i) for i in range(24 * 30)]

        import time

        start_time = time.time()
        heights = model.at(times)
        end_time = time.time()

        self.assertLess(end_time - start_time, 10.0)
        self.assertEqual(len(heights), 24 * 30)

        start_time = time.time()
        extrema = list(
            model.extrema(self.test_time, self.test_time + timedelta(days=30))
        )
        end_time = time.time()

        self.assertLess(end_time - start_time, 30.0)
        self.assertGreater(len(extrema), 0)


if __name__ == "__main__":
    unittest.main()
