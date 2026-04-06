from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytidespy3.constituent as constituent


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_HOLDOUT_HOURS = 24 * 30
DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M"
REPORT_FILENAME = "noaa-benchmark-report.md"
SUMMARY_CSV_FILENAME = "noaa-benchmark-summary.csv"
SUMMARY_JSON_FILENAME = "noaa-benchmark-summary.json"
UTIDE_PIN = "0.3.1"
DEFAULT_DATASET = "hourly"

BENCHMARK_CONSTITUENTS = [
    constituent._M2,
    constituent._S2,
    constituent._K1,
    constituent._O1,
    constituent._N2,
    constituent._P1,
]
UTIDE_CONSTITUENT_NAMES = [item.name for item in BENCHMARK_CONSTITUENTS]


@dataclass(frozen=True)
class BenchmarkStation:
    station_id: str
    name: str
    latitude: float
    regime: str
    setting: str
    rationale: str
    snapshot_paths: tuple[Path, ...]
    holdout_hours: int = DEFAULT_HOLDOUT_HOURS


HOURLY_BENCHMARK_STATIONS = [
    BenchmarkStation(
        station_id="8410140",
        name="Eastport",
        latitude=44.9046,
        regime="semidiurnal",
        setting="outer_coast",
        rationale="High-range outer-coast station used to anchor a cleaner semidiurnal comparison.",
        snapshot_paths=(DATA_DIR / "noaa_8410140_hourly_height_20230101_20230331.json",),
    ),
    BenchmarkStation(
        station_id="8724580",
        name="Key West",
        latitude=24.5557,
        regime="mixed",
        setting="outer_coast",
        rationale="Lower-range southern station to keep the benchmark honest outside the highest-range coastlines.",
        snapshot_paths=(DATA_DIR / "noaa_8724580_hourly_height_20230101_20230331.json",),
    ),
    BenchmarkStation(
        station_id="9414290",
        name="San Francisco",
        latitude=37.8063,
        regime="mixed",
        setting="estuary",
        rationale="Harder estuarine station retained from the existing NOAA regression test lineage.",
        snapshot_paths=(DATA_DIR / "noaa_9414290_hourly_height_20230101_20230331.json",),
    ),
]

SIX_MINUTE_BENCHMARK_STATIONS = [
    BenchmarkStation(
        station_id="8410140",
        name="Eastport",
        latitude=44.9046,
        regime="semidiurnal",
        setting="outer_coast",
        rationale="High-range outer-coast station used to anchor a cleaner semidiurnal comparison.",
        snapshot_paths=(
            DATA_DIR / "noaa_8410140_water_level_20230101_20230131.json",
            DATA_DIR / "noaa_8410140_water_level_20230201_20230228.json",
            DATA_DIR / "noaa_8410140_water_level_20230301_20230331.json",
        ),
    ),
    BenchmarkStation(
        station_id="8724580",
        name="Key West",
        latitude=24.5557,
        regime="mixed",
        setting="outer_coast",
        rationale="Lower-range southern station to keep the benchmark honest outside the highest-range coastlines.",
        snapshot_paths=(
            DATA_DIR / "noaa_8724580_water_level_20230101_20230131.json",
            DATA_DIR / "noaa_8724580_water_level_20230201_20230228.json",
            DATA_DIR / "noaa_8724580_water_level_20230301_20230331.json",
        ),
    ),
    BenchmarkStation(
        station_id="9414290",
        name="San Francisco",
        latitude=37.8063,
        regime="mixed",
        setting="estuary",
        rationale="Harder estuarine station retained from the existing NOAA regression test lineage.",
        snapshot_paths=(
            DATA_DIR / "noaa_9414290_water_level_20230101_20230131.json",
            DATA_DIR / "noaa_9414290_water_level_20230201_20230228.json",
            DATA_DIR / "noaa_9414290_water_level_20230301_20230331.json",
        ),
    ),
]

DATASET_PROFILES = {
    "hourly": HOURLY_BENCHMARK_STATIONS,
    "6-minute": SIX_MINUTE_BENCHMARK_STATIONS,
}

def select_benchmark_stations(station_ids=None, dataset=DEFAULT_DATASET):
    if dataset not in DATASET_PROFILES:
        raise ValueError(f"Unknown benchmark dataset: {dataset}")
    stations = DATASET_PROFILES[dataset]
    if not station_ids:
        return stations
    wanted = set(station_ids)
    selected = [station for station in stations if station.station_id in wanted]
    if len(selected) != len(wanted):
        missing = sorted(wanted - {station.station_id for station in selected})
        raise ValueError(f"Unknown benchmark station ids: {', '.join(missing)}")
    return selected
