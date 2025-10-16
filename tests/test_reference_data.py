import json
import math
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import pytidespy3.constituent as constituent
import pytidespy3.tide as tide


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
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    times = [
        datetime.strptime(entry["t"], "%Y-%m-%d %H:%M")
        for entry in payload["data"]
    ]
    heights = np.asarray([float(entry["v"]) for entry in payload["data"]], dtype=np.float64)
    return times, heights


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
    if target_time <= times[0]:
        return heights[0]
    if target_time >= times[-1]:
        return heights[-1]
    # Binary search for neighbouring timestamps
    left, right = 0, len(times) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if times[mid] <= target_time:
            left = mid
        else:
            right = mid
    span = (times[right] - times[left]).total_seconds()
    if span == 0:
        return heights[left]
    ratio = (target_time - times[left]).total_seconds() / span
    return heights[left] + ratio * (heights[right] - heights[left])


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
        """Model trained on first 48 hours should forecast next 24 hours within tolerance."""
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
        predicted_maxima = [ext for ext in predicted_extrema if ext[2] == "H"]
        predicted_minima = [ext for ext in predicted_extrema if ext[2] == "L"]

        def assert_alignment(observed_extrema, predicted_extrema, kind):
            for obs_time, obs_height in observed_extrema:
                closest_time, closest_height, _ = min(
                    predicted_extrema,
                    key=lambda triple: abs((triple[0] - obs_time).total_seconds()),
                )
                self.assertLessEqual(
                    abs((closest_time - obs_time).total_seconds()),
                    3900,
                    f"{kind} tide prediction differs by more than 1 hour",
                )
                observed_at_predicted = _interpolate_height(self.times, self.heights, closest_time)
                self.assertLess(
                    abs(closest_height - observed_at_predicted),
                    0.32,
                    f"{kind} tide height differs by more than 32 cm at predicted time",
                )
                self.assertLess(
                    abs(observed_at_predicted - obs_height),
                    0.15,
                    f"{kind} observe/extrema mismatch exceeds 15 cm",
                )

        assert_alignment(maxima, predicted_maxima, "H")
        assert_alignment(minima, predicted_minima, "L")


if __name__ == "__main__":
    unittest.main()
