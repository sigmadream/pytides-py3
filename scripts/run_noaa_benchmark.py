#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark_support import run_benchmark, write_benchmark_artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Run the checked-in NOAA benchmark harness and emit Markdown + CSV/JSON artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "benchmark_artifacts"),
        help="Directory to write the benchmark artifacts into.",
    )
    parser.add_argument(
        "--station",
        action="append",
        dest="station_ids",
        help="Optional NOAA station id filter. Repeat the flag to benchmark multiple stations.",
    )
    parser.add_argument(
        "--dataset",
        default="hourly",
        choices=("hourly", "6-minute"),
        help="NOAA snapshot profile to benchmark.",
    )
    args = parser.parse_args()

    results = run_benchmark(station_ids=args.station_ids, dataset=args.dataset)
    paths = write_benchmark_artifacts(results, args.output_dir)

    print(f"Markdown report: {paths['markdown']}")
    print(f"CSV summary: {paths['csv']}")
    print(f"JSON summary: {paths['json']}")


if __name__ == "__main__":
    main()
