import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tests.benchmark_support import (
    BenchmarkDataError,
    collect_extrema_alignment_errors,
    compute_extrema_metrics,
    extract_extrema,
    interpolate_height,
    load_noaa_snapshot,
    load_noaa_snapshots,
    split_train_test,
)


class TestBenchmarkSupport(unittest.TestCase):
    def test_load_noaa_snapshot_rejects_malformed_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "bad.json"
            snapshot.write_text(json.dumps({"metadata": {"id": "x", "name": "Broken"}}), encoding="utf-8")
            with self.assertRaises(BenchmarkDataError):
                load_noaa_snapshot(snapshot)

    def test_split_train_test_rejects_oversized_holdout(self):
        times = [datetime(2023, 1, 1) + timedelta(hours=index) for index in range(8)]
        heights = np.asarray([float(index) for index in range(8)], dtype=np.float64)
        with self.assertRaises(BenchmarkDataError):
            split_train_test(times, heights, holdout_hours=8)

    def test_split_train_test_uses_duration_for_6_minute_series(self):
        times = [datetime(2023, 1, 1) + timedelta(minutes=6 * index) for index in range(480)]
        heights = np.asarray([float(index) for index in range(480)], dtype=np.float64)
        split = split_train_test(times, heights, holdout_hours=24)
        self.assertEqual(len(split["test_times"]), 240)

    def test_load_noaa_snapshots_merges_monthly_files(self):
        merged = load_noaa_snapshots(
            (
                Path(__file__).parent / "data" / "noaa_9414290_water_level_20230101_20230131.json",
                Path(__file__).parent / "data" / "noaa_9414290_water_level_20230201_20230228.json",
            )
        )
        self.assertGreater(len(merged.times), 10000)
        self.assertLess(merged.times[0], merged.times[-1])

    def test_interpolate_height_handles_zero_span(self):
        times = [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0)]
        heights = np.asarray([1.0, 1.2, 2.0], dtype=np.float64)
        self.assertEqual(interpolate_height(times, heights, times[0]), 1.0)

    def test_collect_extrema_alignment_handles_missing_predictions(self):
        times = [datetime(2023, 1, 1) + timedelta(hours=index) for index in range(5)]
        heights = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float64)
        errors = collect_extrema_alignment_errors(times, heights, predicted_extrema=[])
        self.assertEqual(errors["H"], [])
        self.assertEqual(errors["L"], [])

    def test_extract_extrema_returns_empty_for_short_series(self):
        maxima, minima = extract_extrema(
            [datetime(2023, 1, 1), datetime(2023, 1, 1, 1)],
            np.asarray([0.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(maxima, [])
        self.assertEqual(minima, [])

    def test_extract_extrema_coalesces_close_candidates(self):
        times = [datetime(2023, 1, 1) + timedelta(minutes=30 * index) for index in range(9)]
        heights = np.asarray([0.0, 2.0, 1.8, 2.1, 0.0, -1.0, -0.8, -1.1, 0.0], dtype=np.float64)
        maxima, minima = extract_extrema(times, heights)
        self.assertEqual(len(maxima), 1)
        self.assertEqual(len(minima), 1)

    def test_extract_extrema_suppresses_6_minute_shoulder_noise(self):
        times = [datetime(2023, 3, 2) + timedelta(minutes=6 * index) for index in range(69)]
        heights = np.asarray(
            [
                0.456, 0.446, 0.442, 0.445, 0.451, 0.451, 0.443, 0.433, 0.425, 0.418,
                0.413, 0.412, 0.409, 0.398, 0.383, 0.373, 0.369, 0.360, 0.344, 0.327,
                0.322, 0.319, 0.320, 0.310, 0.294, 0.279, 0.269, 0.260, 0.250, 0.243,
                0.240, 0.233, 0.222, 0.213, 0.208, 0.203, 0.190, 0.181, 0.171, 0.167,
                0.164, 0.160, 0.156, 0.139, 0.128, 0.120, 0.111, 0.105, 0.101, 0.099,
                0.101, 0.120, 0.160, 0.240, 0.340, 0.450, 0.560, 0.620, 0.600, 0.590,
                0.610, 0.580, 0.500, 0.420, 0.310, 0.180, 0.080, 0.020, -0.010,
            ],
            dtype=np.float64,
        )
        maxima, minima = extract_extrema(times, heights)
        self.assertEqual(len(maxima), 1)
        self.assertEqual(len(minima), 1)
        self.assertGreater(maxima[0][0], times[50])

    def test_compute_extrema_metrics_reports_p95_time_error(self):
        observed_times = [datetime(2023, 1, 1) + timedelta(hours=index) for index in range(11)]
        observed_heights = np.asarray([0.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0, 1.0, 0.0], dtype=np.float64)
        predicted_heights = np.asarray([0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, -2.0, 1.0, 0.0], dtype=np.float64)
        metrics = compute_extrema_metrics(
            observed_times,
            observed_heights,
            observed_times,
            predicted_heights,
        )
        self.assertEqual(metrics["extrema_pairs"], 5)
        self.assertAlmostEqual(metrics["max_time_error_minutes"], 60.0)
        self.assertIsNotNone(metrics["p95_time_error_minutes"])
        self.assertLessEqual(metrics["p95_time_error_minutes"], metrics["max_time_error_minutes"])
        self.assertGreater(metrics["p95_time_error_minutes"], 0.0)


if __name__ == "__main__":
    unittest.main()
