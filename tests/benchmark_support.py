import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import pytidespy3
import pytidespy3.tide as tide

from tests.benchmark_config import (
    BENCHMARK_CONSTITUENTS,
    DEFAULT_DATASET,
    DEFAULT_TIME_FORMAT,
    REPORT_FILENAME,
    SUMMARY_CSV_FILENAME,
    SUMMARY_JSON_FILENAME,
    UTIDE_CONSTITUENT_NAMES,
    UTIDE_PIN,
    select_benchmark_stations,
)


@dataclass(frozen=True)
class SnapshotSeries:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    times: list
    heights: np.ndarray
    raw_path: Path


class BenchmarkDataError(ValueError):
    pass


def load_noaa_snapshot(path):
    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata")
    data = payload.get("data")
    if not metadata or not data:
        raise BenchmarkDataError(f"{snapshot_path.name} is missing NOAA metadata or data rows.")
    times = [
        datetime.strptime(entry["t"], DEFAULT_TIME_FORMAT)
        for entry in data
    ]
    heights = np.asarray([float(entry["v"]) for entry in data], dtype=np.float64)
    if len(times) == 0:
        raise BenchmarkDataError(f"{snapshot_path.name} has no observations.")
    if len(times) != len(heights):
        raise BenchmarkDataError(f"{snapshot_path.name} has mismatched time and height arrays.")
    return SnapshotSeries(
        station_id=str(metadata["id"]),
        station_name=str(metadata["name"]),
        latitude=float(metadata["lat"]),
        longitude=float(metadata["lon"]),
        times=times,
        heights=heights,
        raw_path=snapshot_path,
    )


def load_noaa_snapshots(paths):
    snapshots = [load_noaa_snapshot(path) for path in paths]
    if not snapshots:
        raise BenchmarkDataError("At least one NOAA snapshot path is required.")
    first = snapshots[0]
    station_ids = {snapshot.station_id for snapshot in snapshots}
    station_names = {snapshot.station_name for snapshot in snapshots}
    if len(station_ids) != 1 or len(station_names) != 1:
        raise BenchmarkDataError("NOAA snapshot bundle mixes multiple stations.")
    merged_points = {}
    for snapshot in snapshots:
        for point_time, point_height in zip(snapshot.times, snapshot.heights):
            merged_points[point_time] = float(point_height)
    merged_times = sorted(merged_points)
    merged_heights = np.asarray([merged_points[item] for item in merged_times], dtype=np.float64)
    return SnapshotSeries(
        station_id=first.station_id,
        station_name=first.station_name,
        latitude=first.latitude,
        longitude=first.longitude,
        times=merged_times,
        heights=merged_heights,
        raw_path=first.raw_path,
    )


def extract_extrema(times, heights):
    maxima, minima = [], []
    if len(heights) < 3:
        return maxima, minima
    smoothed_heights = _smooth_heights(heights, _extrema_smoothing_window(times))
    for idx in range(1, len(heights) - 1):
        if smoothed_heights[idx] > smoothed_heights[idx - 1] and smoothed_heights[idx] > smoothed_heights[idx + 1]:
            maxima.append((times[idx], float(heights[idx])))
        elif smoothed_heights[idx] < smoothed_heights[idx - 1] and smoothed_heights[idx] < smoothed_heights[idx + 1]:
            minima.append((times[idx], float(heights[idx])))
    min_separation_hours = _extrema_min_separation_hours(times)
    return (
        _coalesce_extrema(maxima, is_maxima=True, min_separation_hours=min_separation_hours),
        _coalesce_extrema(minima, is_maxima=False, min_separation_hours=min_separation_hours),
    )


def discrete_extrema(times, heights):
    maxima, minima = extract_extrema(times, heights)
    return [(item[0], item[1], "H") for item in maxima] + [(item[0], item[1], "L") for item in minima]


def interpolate_height(times, heights, target_time):
    if target_time <= times[0]:
        return float(heights[0])
    if target_time >= times[-1]:
        return float(heights[-1])
    left, right = 0, len(times) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if times[mid] <= target_time:
            left = mid
        else:
            right = mid
    span = (times[right] - times[left]).total_seconds()
    if span == 0:
        return float(heights[left])
    ratio = (target_time - times[left]).total_seconds() / span
    return float(heights[left] + ratio * (heights[right] - heights[left]))


def collect_extrema_alignment_errors(times, heights, predicted_extrema):
    maxima, minima = extract_extrema(times, heights)
    predicted_maxima = [item for item in predicted_extrema if item[2] == "H"]
    predicted_minima = [item for item in predicted_extrema if item[2] == "L"]
    return {
        "H": _collect_alignment_errors(maxima, predicted_maxima, times, heights),
        "L": _collect_alignment_errors(minima, predicted_minima, times, heights),
    }


def _collect_alignment_errors(observed_extrema, predicted_extrema, times, heights):
    if not observed_extrema or not predicted_extrema:
        return []
    errors = []
    for obs_time, obs_height in observed_extrema:
        closest_time, closest_height, _ = min(
            predicted_extrema,
            key=lambda triple: abs((triple[0] - obs_time).total_seconds()),
        )
        observed_at_predicted = interpolate_height(times, heights, closest_time)
        errors.append(
            {
                "observed_time": obs_time.isoformat(),
                "predicted_time": closest_time.isoformat(),
                "time_error_seconds": abs((closest_time - obs_time).total_seconds()),
                "predicted_height_error": abs(float(closest_height) - observed_at_predicted),
                "observed_height_error": abs(observed_at_predicted - obs_height),
            }
        )
    return errors


def split_train_test(times, heights, holdout_hours):
    if holdout_hours <= 0:
        raise BenchmarkDataError("holdout_hours must be positive.")
    if len(times) < 3:
        raise BenchmarkDataError("Need at least three samples to split benchmark data.")
    intervals = [
        (times[index + 1] - times[index]).total_seconds()
        for index in range(len(times) - 1)
    ]
    median_interval_seconds = sorted(intervals)[len(intervals) // 2]
    cutoff_time = times[-1] - timedelta(hours=holdout_hours) + timedelta(seconds=median_interval_seconds)
    split_index = next((index for index, point_time in enumerate(times) if point_time >= cutoff_time), len(times))
    if split_index <= 1 or split_index >= len(times):
        raise BenchmarkDataError(
            f"holdout_hours={holdout_hours} leaves an invalid train/test split for {len(times)} samples."
        )
    return {
        "train_times": times[:split_index],
        "train_heights": heights[:split_index],
        "test_times": times[split_index:],
        "test_heights": heights[split_index:],
    }


def compute_basic_metrics(predictions, observed):
    if len(predictions) != len(observed):
        raise BenchmarkDataError("prediction and observation lengths differ.")
    if len(predictions) == 0:
        raise BenchmarkDataError("benchmark window has no prediction samples.")
    predictions = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if not np.all(np.isfinite(predictions)):
        raise BenchmarkDataError("predictions contain non-finite values.")
    rmse = math.sqrt(np.mean((predictions - observed) ** 2))
    mae = float(np.mean(np.abs(predictions - observed)))
    bias = float(np.mean(predictions - observed))
    correlation = None
    if len(predictions) > 1 and not np.allclose(predictions, predictions[0]) and not np.allclose(observed, observed[0]):
        correlation = float(np.corrcoef(predictions, observed)[0, 1])
    return {
        "rmse": float(rmse),
        "mae": mae,
        "bias": bias,
        "correlation": correlation,
    }


def compute_extrema_metrics(observed_times, observed_heights, predicted_times, predicted_heights):
    predicted_extrema = discrete_extrema(predicted_times, predicted_heights)
    errors = collect_extrema_alignment_errors(observed_times, observed_heights, predicted_extrema)
    flattened = errors["H"] + errors["L"]
    if not flattened:
        return {
            "extrema_pairs": 0,
            "max_time_error_minutes": None,
            "p95_time_error_minutes": None,
            "max_predicted_height_error": None,
            "max_observed_height_error": None,
        }
    time_error_minutes = [item["time_error_seconds"] / 60.0 for item in flattened]
    return {
        "extrema_pairs": len(flattened),
        "max_time_error_minutes": max(time_error_minutes),
        "p95_time_error_minutes": _percentile(time_error_minutes, 0.95),
        "max_predicted_height_error": max(item["predicted_height_error"] for item in flattened),
        "max_observed_height_error": max(item["observed_height_error"] for item in flattened),
    }


def compute_engine_metrics(test_times, test_heights, predictions):
    metrics = compute_basic_metrics(predictions, test_heights)
    metrics.update(compute_extrema_metrics(test_times, test_heights, test_times, predictions))
    return metrics


def run_pytides_engine(split_payload):
    model = tide.Tide.decompose(
        heights=split_payload["train_heights"],
        t=split_payload["train_times"],
        constituents=BENCHMARK_CONSTITUENTS,
        n_period=0,
    )
    predictions = np.asarray(model.at(split_payload["test_times"]), dtype=np.float64)
    return {
        "engine": "pytides-py3",
        "status": "ok",
        "version": pytidespy3.__version__,
        "metrics": compute_engine_metrics(
            split_payload["test_times"],
            split_payload["test_heights"],
            predictions,
        ),
        "error": None,
    }


def load_utide_module():
    try:
        return importlib.import_module("utide")
    except ModuleNotFoundError as exc:
        raise BenchmarkDataError(
            f"UTide is not installed. Install utide=={UTIDE_PIN} to enable the comparison runner."
        ) from exc


def run_utide_engine(split_payload, latitude, utide_module=None):
    utide = utide_module if utide_module is not None else load_utide_module()
    train_times = np.asarray(split_payload["train_times"], dtype="datetime64[ns]")
    train_heights = np.asarray(split_payload["train_heights"], dtype=np.float64)
    test_times = np.asarray(split_payload["test_times"], dtype="datetime64[ns]")
    solve_result = utide.solve(
        train_times,
        train_heights,
        lat=latitude,
        constit=UTIDE_CONSTITUENT_NAMES,
        trend=False,
        conf_int="none",
        verbose=False,
    )
    reconstruction = utide.reconstruct(
        test_times,
        solve_result,
        verbose=False,
        constit=UTIDE_CONSTITUENT_NAMES,
    )
    predictions = np.asarray(reconstruction.h, dtype=np.float64)
    return {
        "engine": "UTide",
        "status": "ok",
        "version": getattr(utide, "__version__", UTIDE_PIN),
        "metrics": compute_engine_metrics(
            split_payload["test_times"],
            split_payload["test_heights"],
            predictions,
        ),
        "error": None,
    }


def run_station_benchmark(station, utide_module=None):
    series = load_noaa_snapshots(station.snapshot_paths)
    split_payload = split_train_test(series.times, series.heights, station.holdout_hours)
    station_result = {
        "station_id": station.station_id,
        "station_name": station.name,
        "regime": station.regime,
        "setting": station.setting,
        "rationale": station.rationale,
        "snapshot_paths": [str(Path(path).resolve()) for path in station.snapshot_paths],
        "train_samples": len(split_payload["train_times"]),
        "test_samples": len(split_payload["test_times"]),
        "engines": [],
    }
    for engine_name, runner in (
        ("pytides-py3", lambda: run_pytides_engine(split_payload)),
        ("UTide", lambda: run_utide_engine(split_payload, station.latitude, utide_module=utide_module)),
    ):
        try:
            station_result["engines"].append(runner())
        except Exception as exc:  # noqa: BLE001
            station_result["engines"].append(
                {
                    "engine": engine_name,
                    "status": "failed",
                    "version": None,
                    "metrics": None,
                    "error": str(exc),
                }
            )
    return station_result


def run_benchmark(station_ids=None, utide_module=None, dataset=DEFAULT_DATASET):
    results = []
    for station in select_benchmark_stations(station_ids=station_ids, dataset=dataset):
        try:
            results.append(run_station_benchmark(station, utide_module=utide_module))
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "station_id": station.station_id,
                    "station_name": station.name,
                    "regime": station.regime,
                    "setting": station.setting,
                    "rationale": station.rationale,
                    "snapshot_paths": [str(Path(path).resolve()) for path in station.snapshot_paths],
                    "train_samples": 0,
                    "test_samples": 0,
                    "engines": [
                        {
                            "engine": "pytides-py3",
                            "status": "failed",
                            "version": pytidespy3.__version__,
                            "metrics": None,
                            "error": str(exc),
                        },
                        {
                            "engine": "UTide",
                            "status": "failed",
                            "version": None,
                            "metrics": None,
                            "error": "Station setup failed before UTide could run.",
                        },
                    ],
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": dataset,
        "stations": results,
    }


def _summary_rows(results):
    rows = []
    for station in results["stations"]:
        for engine in station["engines"]:
            metrics = engine["metrics"] or {}
            rows.append(
                {
                    "station_id": station["station_id"],
                    "station_name": station["station_name"],
                    "regime": station["regime"],
                    "setting": station["setting"],
                    "engine": engine["engine"],
                    "status": engine["status"],
                    "version": engine["version"] or "",
                    "rmse": _format_optional(metrics.get("rmse")),
                    "mae": _format_optional(metrics.get("mae")),
                    "bias": _format_optional(metrics.get("bias")),
                    "correlation": _format_optional(metrics.get("correlation")),
                    "extrema_pairs": metrics.get("extrema_pairs", ""),
                    "max_time_error_minutes": _format_optional(metrics.get("max_time_error_minutes")),
                    "p95_time_error_minutes": _format_optional(metrics.get("p95_time_error_minutes")),
                    "max_predicted_height_error": _format_optional(metrics.get("max_predicted_height_error")),
                    "max_observed_height_error": _format_optional(metrics.get("max_observed_height_error")),
                    "error": engine["error"] or "",
                }
            )
    return rows


def _summarize_results(results):
    rows = _summary_rows(results)
    by_engine = {}
    for row in rows:
        engine_summary = by_engine.setdefault(
            row["engine"],
            {
                "engine": row["engine"],
                "ok_stations": 0,
                "failed_stations": 0,
                "avg_rmse": None,
                "avg_mae": None,
                "avg_bias": None,
                "avg_correlation": None,
            },
        )
        if row["status"] == "ok":
            engine_summary["ok_stations"] += 1
        else:
            engine_summary["failed_stations"] += 1

    for engine_summary in by_engine.values():
        ok_rows = [
            row
            for row in rows
            if row["engine"] == engine_summary["engine"] and row["status"] == "ok"
        ]
        if ok_rows:
            engine_summary["avg_rmse"] = _average_numeric(ok_rows, "rmse")
            engine_summary["avg_mae"] = _average_numeric(ok_rows, "mae")
            engine_summary["avg_bias"] = _average_numeric(ok_rows, "bias")
            engine_summary["avg_correlation"] = _average_numeric(ok_rows, "correlation")

    station_deltas = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["station_id"], {})[row["engine"]] = row
    for station_id, engines in grouped.items():
        if {"pytides-py3", "UTide"} - set(engines):
            continue
        pytides_row = engines["pytides-py3"]
        utide_row = engines["UTide"]
        if pytides_row["status"] != "ok" or utide_row["status"] != "ok":
            continue
        rmse_delta = _numeric_delta(pytides_row["rmse"], utide_row["rmse"])
        mae_delta = _numeric_delta(pytides_row["mae"], utide_row["mae"])
        corr_delta = _numeric_delta(pytides_row["correlation"], utide_row["correlation"])
        station_deltas.append(
            {
                "station_id": station_id,
                "station_name": pytides_row["station_name"],
                "rmse_delta": rmse_delta,
                "mae_delta": mae_delta,
                "correlation_delta": corr_delta,
                "winner": _winner_label(rmse_delta),
            }
        )

    hardest_station = None
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        hardest_row = max(ok_rows, key=lambda row: float(row["rmse"]))
        hardest_station = {
            "station_id": hardest_row["station_id"],
            "station_name": hardest_row["station_name"],
            "engine": hardest_row["engine"],
            "rmse": float(hardest_row["rmse"]),
        }

    findings = []
    pytides_summary = next((item for item in by_engine.values() if item["engine"] == "pytides-py3"), None)
    utide_summary = next((item for item in by_engine.values() if item["engine"] == "UTide"), None)
    if pytides_summary and utide_summary and pytides_summary["avg_rmse"] is not None and utide_summary["avg_rmse"] is not None:
        avg_rmse_delta = pytides_summary["avg_rmse"] - utide_summary["avg_rmse"]
        findings.append(
            "Overall RMSE delta (pytides-py3 - UTide): "
            f"{avg_rmse_delta:+.4f} across {min(pytides_summary['ok_stations'], utide_summary['ok_stations'])} paired stations."
        )
        if abs(avg_rmse_delta) < 0.005:
            findings.append("The current benchmark reads as a near-tie on average RMSE, which supports a reproducibility claim more than a superiority claim.")

    large_extrema_rows = [
        row for row in rows
        if row["status"] == "ok"
        and row["max_time_error_minutes"] not in ("", None)
        and float(row["max_time_error_minutes"]) >= 120.0
    ]
    if large_extrema_rows:
        station_list = ", ".join(
            f"{row['station_name']} ({row['engine']})" for row in large_extrema_rows
        )
        findings.append(
            "At least one station shows large extrema timing error (>= 120 minutes). "
            f"Current examples: {station_list}. This likely reflects a strict discrete-extrema alignment metric, not just model quality."
        )
    elevated_p95_rows = [
        row for row in rows
        if row["status"] == "ok"
        and row["p95_time_error_minutes"] not in ("", None)
        and float(row["p95_time_error_minutes"]) >= 60.0
    ]
    if elevated_p95_rows:
        station_list = ", ".join(
            f"{row['station_name']} ({row['engine']}, p95={row['p95_time_error_minutes']} min)"
            for row in elevated_p95_rows
        )
        findings.append(
            "Some stations still show elevated p95 extrema timing error (>= 60 minutes). "
            f"Current examples: {station_list}."
        )
    else:
        stable_p95_rows = [
            row for row in rows
            if row["status"] == "ok" and row["p95_time_error_minutes"] not in ("", None)
        ]
        if stable_p95_rows:
            station_list = ", ".join(
                f"{row['station_name']} ({row['engine']}, p95={row['p95_time_error_minutes']} min)"
                for row in sorted(stable_p95_rows, key=lambda row: float(row["p95_time_error_minutes"]), reverse=True)[:3]
            )
            findings.append(
                "The p95 extrema timing error stays below 60 minutes across the current benchmark set. "
                f"Current highest examples: {station_list}."
            )

    return {
        "rows": rows,
        "engine_summary": list(by_engine.values()),
        "station_deltas": station_deltas,
        "hardest_station": hardest_station,
        "findings": findings,
    }


def write_benchmark_artifacts(results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / REPORT_FILENAME
    csv_path = output_path / SUMMARY_CSV_FILENAME
    json_path = output_path / SUMMARY_JSON_FILENAME

    summary = _summarize_results(results)
    rows = summary["rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_payload = dict(results)
    json_payload["aggregate"] = {
        "engine_summary": summary["engine_summary"],
        "station_deltas": summary["station_deltas"],
        "hardest_station": summary["hardest_station"],
        "findings": summary["findings"],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    markdown_path.write_text(render_markdown_report(results), encoding="utf-8")

    return {
        "markdown": str(markdown_path),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def render_markdown_report(results):
    summary = _summarize_results(results)
    dataset_label = results.get("dataset", DEFAULT_DATASET)
    source_phrase = "checked-in NOAA hourly height snapshots" if dataset_label == "hourly" else "checked-in NOAA 6-minute water-level snapshots"
    lines = [
        "# NOAA Benchmark Report",
        "",
        f"Generated: {results['generated_at']}",
        "",
        f"Dataset: {results.get('dataset', DEFAULT_DATASET)}",
        "",
        f"This benchmark uses {source_phrase}, a shared train/test split,",
        "and identical summary metrics for `pytides-py3` and `UTide`.",
        "",
        "## Overall Summary",
        "",
        "| Engine | OK stations | Failed stations | Avg RMSE | Avg MAE | Avg Bias | Avg Corr |",
        "|--------|-------------|-----------------|----------|---------|----------|----------|",
    ]
    for engine_summary in summary["engine_summary"]:
        lines.append(
            "| {engine} | {ok_stations} | {failed_stations} | {avg_rmse} | {avg_mae} | {avg_bias} | {avg_correlation} |".format(
                engine=engine_summary["engine"],
                ok_stations=engine_summary["ok_stations"],
                failed_stations=engine_summary["failed_stations"],
                avg_rmse=_format_optional(engine_summary["avg_rmse"]),
                avg_mae=_format_optional(engine_summary["avg_mae"]),
                avg_bias=_format_optional(engine_summary["avg_bias"]),
                avg_correlation=_format_optional(engine_summary["avg_correlation"]),
            )
        )

    lines.extend(["", "## Pairwise Deltas", ""])
    if not summary["station_deltas"]:
        lines.append("No station has a complete `pytides-py3` + `UTide` pair to compare.")
    else:
        lines.extend(
            [
                "| Station | RMSE delta (pytides - UTide) | MAE delta | Corr delta | Winner |",
                "|--------|-------------------------------|-----------|------------|--------|",
            ]
        )
        for delta in summary["station_deltas"]:
            lines.append(
                "| {station_name} ({station_id}) | {rmse_delta} | {mae_delta} | {correlation_delta} | {winner} |".format(
                    station_name=delta["station_name"],
                    station_id=delta["station_id"],
                    rmse_delta=_format_optional(delta["rmse_delta"]),
                    mae_delta=_format_optional(delta["mae_delta"]),
                    correlation_delta=_format_optional(delta["correlation_delta"]),
                    winner=delta["winner"],
                )
            )

    if summary["hardest_station"] is not None:
        hardest_station = summary["hardest_station"]
        lines.extend(
            [
                "",
                f"Hardest benchmark slice so far: {hardest_station['station_name']} ({hardest_station['station_id']})"
                f" via {hardest_station['engine']} with RMSE {hardest_station['rmse']:.4f}.",
            ]
        )

    lines.extend(["", "## Findings", ""])
    if summary["findings"]:
        for finding in summary["findings"]:
            lines.append(f"- {finding}")
    else:
        lines.append("No additional benchmark findings generated.")

    lines.extend([
        "",
        "## Station Summary",
        "",
        "| Station | Regime | Setting | Engine | Status | RMSE | MAE | Bias | Corr | Max extrema time err (min) | p95 extrema time err (min) | Error |",
        "|--------|--------|---------|--------|--------|------|-----|------|------|----------------------------|----------------------------|-------|",
    ])
    for row in summary["rows"]:
        lines.append(
            "| {station_name} ({station_id}) | {regime} | {setting} | {engine} | {status} | {rmse} | {mae} | {bias} | {correlation} | {max_time_error_minutes} | {p95_time_error_minutes} | {error} |".format(
                **row
            )
        )

    failures = [
        (station["station_name"], station["station_id"], engine)
        for station in results["stations"]
        for engine in station["engines"]
        if engine["status"] != "ok"
    ]
    lines.extend(["", "## Failure Details", ""])
    if not failures:
        lines.append("No station-level failures recorded.")
    else:
        for station_name, station_id, engine in failures:
            lines.append(
                f"- {station_name} ({station_id}) / {engine['engine']}: {engine['error']}"
            )

    lines.extend(["", "## Station Rationale", ""])
    for station in results["stations"]:
        lines.append(
            f"- {station['station_name']} ({station['station_id']}): {station['rationale']}"
        )
    lines.append("")
    return "\n".join(lines)


def _format_optional(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _average_numeric(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not values:
        return None
    return sum(values) / len(values)


def _winner_label(rmse_delta):
    if rmse_delta is None:
        return "n/a"
    if abs(rmse_delta) < 0.001:
        return "near-tie"
    if rmse_delta < 0:
        return "pytides-py3"
    return "UTide"


def _numeric_delta(left, right):
    if left in ("", None) or right in ("", None):
        return None
    return float(left) - float(right)


def _percentile(values, quantile):
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def _coalesce_extrema(extrema, is_maxima, min_separation_hours=3.0):
    if not extrema:
        return []
    coalesced = [extrema[0]]
    min_separation_seconds = min_separation_hours * 3600.0
    for candidate_time, candidate_height in extrema[1:]:
        last_time, last_height = coalesced[-1]
        if (candidate_time - last_time).total_seconds() < min_separation_seconds:
            if (is_maxima and candidate_height > last_height) or ((not is_maxima) and candidate_height < last_height):
                coalesced[-1] = (candidate_time, candidate_height)
        else:
            coalesced.append((candidate_time, candidate_height))
    return coalesced


def _median_interval_seconds(times):
    if len(times) < 2:
        return None
    intervals = sorted(
        (times[index + 1] - times[index]).total_seconds()
        for index in range(len(times) - 1)
    )
    return intervals[len(intervals) // 2]


def _extrema_smoothing_window(times):
    median_interval_seconds = _median_interval_seconds(times)
    if median_interval_seconds is None:
        return 1
    if median_interval_seconds <= 15 * 60:
        return 25
    return 1


def _extrema_min_separation_hours(times):
    median_interval_seconds = _median_interval_seconds(times)
    if median_interval_seconds is None:
        return 3.0
    if median_interval_seconds <= 15 * 60:
        return 7.0
    return 3.0


def _smooth_heights(heights, window):
    if window <= 1:
        return np.asarray(heights, dtype=np.float64)
    pad = window // 2
    padded = np.pad(np.asarray(heights, dtype=np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")
