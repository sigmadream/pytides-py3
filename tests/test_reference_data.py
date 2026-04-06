import json
import math
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import pytidespy3.constituent as constituent
import pytidespy3.tide as tide
from tests.benchmark_support import (
    collect_extrema_alignment_errors,
    interpolate_height,
    load_noaa_snapshot,
)


DATA_PATH = Path(__file__).parent / "data" / "noaa_9414290_hourly_height_20230101_20230108.json"
NOAA_MAJOR_CONSTITUENTS = [
    constituent._M2,
    constituent._S2,
    constituent._K1,
    constituent._O1,
    constituent._N2,
    constituent._P1,
]


def _load_noaa_dataset():
    snapshot = load_noaa_snapshot(DATA_PATH)
    return snapshot.times, snapshot.heights


def _extract_extrema(times, heights):
    maxima, minima = [], []
    for idx in range(1, len(heights) - 1):
        if heights[idx] > heights[idx - 1] and heights[idx] > heights[idx + 1]:
            maxima.append((times[idx], heights[idx]))
        elif heights[idx] < heights[idx - 1] and heights[idx] < heights[idx + 1]:
            minima.append((times[idx], heights[idx]))
    return maxima, minima


def _interpolate_height(times, heights, target_time):
    """Linear interpolation helper for irregular evaluations."""
    return interpolate_height(times, heights, target_time)


class TestNoaaCrossCheck(unittest.TestCase):
    """Regression tests against NOAA hourly height observations."""

    @classmethod
    def setUpClass(cls):
        cls.times, cls.heights = _load_noaa_dataset()

    def test_decomposition_rmse(self):
        """Fitting full dataset should achieve low RMSE when reconstructing."""
        model = tide.Tide.decompose(
            heights=self.heights,
            t=self.times,
            constituents=NOAA_MAJOR_CONSTITUENTS,
            n_period=0,
        )
        reconstructed = model.at(self.times)
        rmse = math.sqrt(np.mean((reconstructed - self.heights) ** 2))
        self.assertLess(rmse, 0.18, "RMSE exceeds 18 cm on NOAA dataset")
        bias = float(np.mean(reconstructed - self.heights))
        self.assertLess(abs(bias), 0.05, "Mean bias exceeds 5 cm on NOAA dataset")
        corr = np.corrcoef(reconstructed, self.heights)[0, 1]
        self.assertGreater(corr, 0.96, "Correlation with NOAA observations below 0.96")

    def test_holdout_forecast_accuracy(self):
        """Model trained on first ~6 days should forecast last 48 hours within tolerance."""
        split_index = len(self.times) - 48
        train_times = self.times[:split_index]
        train_heights = self.heights[:split_index]
        test_times = self.times[split_index:]
        test_heights = self.heights[split_index:]

        model = tide.Tide.decompose(
            heights=train_heights,
            t=train_times,
            constituents=NOAA_MAJOR_CONSTITUENTS,
            n_period=0,
        )
        predictions = model.at(test_times)
        mae = np.mean(np.abs(predictions - test_heights))
        self.assertLess(mae, 0.45, "MAE exceeds 45 cm on NOAA hold-out segment")
        rmse = math.sqrt(np.mean((predictions - test_heights) ** 2))
        self.assertLess(rmse, 0.55, "RMSE exceeds 55 cm on NOAA hold-out segment")
        corr = np.corrcoef(predictions, test_heights)[0, 1]
        self.assertGreater(corr, 0.9, "Correlation on hold-out segment below 0.90")

    def test_extrema_alignment(self):
        """Predicted high/low tides should align with observed extrema within tolerance."""
        model = tide.Tide.decompose(
            heights=self.heights,
            t=self.times,
            constituents=NOAA_MAJOR_CONSTITUENTS,
            n_period=0,
        )
        maxima, minima = _extract_extrema(self.times, self.heights)
        predicted_extrema = list(model.extrema(self.times[0], self.times[-1]))
        errors = collect_extrema_alignment_errors(self.times, self.heights, predicted_extrema)

        for kind in ("H", "L"):
            for item in errors[kind]:
                self.assertLessEqual(
                    item["time_error_seconds"],
                    3900,
                    f"{kind} tide prediction differs by more than 65 minutes",
                )
                self.assertLess(
                    item["predicted_height_error"],
                    0.32,
                    f"{kind} tide height differs by more than 32 cm at predicted time",
                )
                self.assertLess(
                    item["observed_height_error"],
                    0.15,
                    f"{kind} observe/extrema mismatch exceeds 15 cm",
                )


if __name__ == "__main__":
    unittest.main()
