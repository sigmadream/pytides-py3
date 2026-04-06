#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark_compare import write_resolution_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare hourly and 6-minute NOAA benchmark artifacts and emit a Markdown summary."
    )
    parser.add_argument(
        "--hourly-summary",
        default=str(PROJECT_ROOT / "benchmark_artifacts" / "noaa-benchmark-summary.csv"),
        help="Path to the hourly benchmark CSV summary.",
    )
    parser.add_argument(
        "--six-minute-summary",
        default=str(PROJECT_ROOT / "benchmark_artifacts_6min" / "noaa-benchmark-summary.csv"),
        help="Path to the 6-minute benchmark CSV summary.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "benchmark_artifacts" / "benchmark-resolution-comparison.md"),
        help="Path to write the comparison Markdown report to.",
    )
    args = parser.parse_args()

    output_path = write_resolution_comparison(
        hourly_summary_path=args.hourly_summary,
        six_minute_summary_path=args.six_minute_summary,
        output_path=args.output,
    )
    print(f"Comparison report: {output_path}")


if __name__ == "__main__":
    main()
