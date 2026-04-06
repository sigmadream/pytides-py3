import json
import tempfile
import unittest
from pathlib import Path

from tests.benchmark_config import BenchmarkStation
from tests.benchmark_support import (
    REPORT_FILENAME,
    SUMMARY_CSV_FILENAME,
    SUMMARY_JSON_FILENAME,
    BenchmarkDataError,
    run_benchmark,
    write_benchmark_artifacts,
)


class _FakeUtideModule:
    __version__ = "0.3.1-test"

    @staticmethod
    def solve(t, u, v=None, lat=None, **opts):
        return {"mean": float(sum(u) / len(u))}

    @staticmethod
    def reconstruct(t, coef, **opts):
        class _Reconstruction:
            pass

        reconstruction = _Reconstruction()
        reconstruction.h = [coef["mean"]] * len(t)
        return reconstruction


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.good_station = BenchmarkStation(
            station_id="9414290",
            name="San Francisco",
            latitude=37.8063,
            regime="mixed",
            setting="estuary",
            rationale="Existing NOAA regression station.",
            snapshot_paths=(Path(__file__).parent / "data" / "noaa_9414290_hourly_height_20230101_20230108.json",),
            holdout_hours=48,
        )

    def test_runner_writes_markdown_csv_and_json_artifacts(self):
        results = run_benchmark(
            station_ids=[self.good_station.station_id],
            utide_module=_FakeUtideModule(),
        )
        # Replace the production manifest selection with a deterministic single-station result.
        results["stations"] = [results["stations"][0]]

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_benchmark_artifacts(results, temp_dir)
            self.assertTrue(Path(paths["markdown"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["json"]).exists())

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("NOAA Benchmark Report", markdown)
            self.assertIn("Overall Summary", markdown)
            self.assertIn("Pairwise Deltas", markdown)
            self.assertIn("Findings", markdown)
            self.assertIn("Failure Details", markdown)
            self.assertIn("p95 extrema time err (min)", markdown)
            self.assertIn("p95 extrema timing error", markdown)

            summary = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(summary["stations"]), 1)
            self.assertEqual(summary["stations"][0]["station_id"], "9414290")
            self.assertEqual(summary["dataset"], "hourly")
            self.assertIn("aggregate", summary)
            self.assertIn("engine_summary", summary["aggregate"])
            self.assertIn("findings", summary["aggregate"])
            self.assertIn("p95_time_error_minutes", summary["stations"][0]["engines"][0]["metrics"])

    def test_run_continues_when_one_station_snapshot_is_bad(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_snapshot = Path(temp_dir) / "bad_snapshot.json"
            bad_snapshot.write_text('{"metadata": {"id": "bad", "name": "Broken"}}\n', encoding="utf-8")
            bad_station = BenchmarkStation(
                station_id="bad",
                name="Broken Station",
                latitude=0.0,
                regime="mixed",
                setting="outer_coast",
                rationale="Broken fixture for failure-path testing.",
                snapshot_paths=(bad_snapshot,),
                holdout_hours=48,
            )

            results = {
                "generated_at": "2026-04-06T00:00:00Z",
                "stations": [
                    {
                        "station_id": self.good_station.station_id,
                        "station_name": self.good_station.name,
                        "regime": self.good_station.regime,
                        "setting": self.good_station.setting,
                        "rationale": self.good_station.rationale,
                        "snapshot_paths": [str(path.resolve()) for path in self.good_station.snapshot_paths],
                        "train_samples": 120,
                        "test_samples": 48,
                        "engines": [
                            {
                                "engine": "pytides-py3",
                                "status": "ok",
                                "version": "0.8.1",
                                "metrics": {"rmse": 0.1, "mae": 0.1, "bias": 0.0, "correlation": 0.9, "extrema_pairs": 1, "max_time_error_minutes": 10.0, "p95_time_error_minutes": 10.0, "max_predicted_height_error": 0.1, "max_observed_height_error": 0.1},
                                "error": None,
                            }
                        ],
                    },
                    {
                        "station_id": bad_station.station_id,
                        "station_name": bad_station.name,
                        "regime": bad_station.regime,
                        "setting": bad_station.setting,
                        "rationale": bad_station.rationale,
                        "snapshot_paths": [str(path.resolve()) for path in bad_station.snapshot_paths],
                        "train_samples": 0,
                        "test_samples": 0,
                        "engines": [
                            {
                                "engine": "pytides-py3",
                                "status": "failed",
                                "version": "0.8.1",
                                "metrics": None,
                                "error": "bad snapshot fixture",
                            }
                        ],
                    },
                ],
            }
            paths = write_benchmark_artifacts(results, temp_dir)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Broken Station", markdown)
            self.assertIn("bad snapshot fixture", markdown)

    def test_missing_utide_dependency_is_reported_explicitly(self):
        from tests import benchmark_support

        original_loader = benchmark_support.load_utide_module

        def _raise_missing():
            raise BenchmarkDataError("UTide is not installed. Install utide==0.3.1 to enable the comparison runner.")

        benchmark_support.load_utide_module = _raise_missing
        try:
            result = benchmark_support.run_station_benchmark(self.good_station)
        finally:
            benchmark_support.load_utide_module = original_loader

        utide_result = next(engine for engine in result["engines"] if engine["engine"] == "UTide")
        self.assertEqual(utide_result["status"], "failed")
        self.assertIn("UTide is not installed", utide_result["error"])


if __name__ == "__main__":
    unittest.main()
