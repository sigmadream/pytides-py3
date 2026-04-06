import tempfile
import unittest
from pathlib import Path

from tests.benchmark_compare import write_resolution_comparison


class TestBenchmarkCompare(unittest.TestCase):
    def test_write_resolution_comparison_renders_expected_sections(self):
        hourly_csv = """station_id,station_name,regime,setting,engine,status,version,rmse,mae,bias,correlation,extrema_pairs,max_time_error_minutes,p95_time_error_minutes,max_predicted_height_error,max_observed_height_error,error
8410140,Eastport,semidiurnal,outer_coast,pytides-py3,ok,0.8.1,0.2500,0.2000,-0.0700,0.9900,100,60.0000,60.0000,0.7000,0.3000,
8410140,Eastport,semidiurnal,outer_coast,UTide,ok,0.3.1,0.2480,0.1980,-0.0680,0.9910,100,60.0000,60.0000,0.6900,0.2000,
"""
        six_minute_csv = """station_id,station_name,regime,setting,engine,status,version,rmse,mae,bias,correlation,extrema_pairs,max_time_error_minutes,p95_time_error_minutes,max_predicted_height_error,max_observed_height_error,error
8410140,Eastport,semidiurnal,outer_coast,pytides-py3,ok,0.8.1,0.2510,0.2010,-0.0710,0.9900,100,24.0000,24.0000,0.7100,0.0900,
8410140,Eastport,semidiurnal,outer_coast,UTide,ok,0.3.1,0.2485,0.1985,-0.0685,0.9910,100,24.0000,18.0000,0.7000,0.0900,
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hourly_path = temp_path / "hourly.csv"
            six_minute_path = temp_path / "6min.csv"
            output_path = temp_path / "comparison.md"
            hourly_path.write_text(hourly_csv, encoding="utf-8")
            six_minute_path.write_text(six_minute_csv, encoding="utf-8")

            write_resolution_comparison(hourly_path, six_minute_path, output_path)

            report = output_path.read_text(encoding="utf-8")
            self.assertIn("NOAA Benchmark Resolution Comparison", report)
            self.assertIn("Engine Averages", report)
            self.assertIn("Station Deltas", report)
            self.assertIn("RMSE delta (6-minute - hourly)", report)
            self.assertIn("Average p95 extrema timing error improves", report)


if __name__ == "__main__":
    unittest.main()
