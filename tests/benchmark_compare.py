import csv
from pathlib import Path


def load_benchmark_summary(path):
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_resolution_comparison(hourly_rows, six_minute_rows):
    hourly_by_key = {(row["station_id"], row["engine"]): row for row in hourly_rows}
    six_minute_by_key = {(row["station_id"], row["engine"]): row for row in six_minute_rows}
    shared_keys = sorted(set(hourly_by_key) & set(six_minute_by_key))

    lines = [
        "# NOAA Benchmark Resolution Comparison",
        "",
        "This report compares the checked-in `hourly` and `6-minute` NOAA benchmark artifacts.",
        "Positive deltas mean the `6-minute` benchmark is stricter on that metric.",
        "",
        "## Engine Averages",
        "",
        "| Engine | Hourly Avg RMSE | 6-minute Avg RMSE | Hourly Avg p95 extrema err (min) | 6-minute Avg p95 extrema err (min) |",
        "|--------|-----------------|-------------------|----------------------------------|------------------------------------|",
    ]

    for engine in ("pytides-py3", "UTide"):
        hourly_engine_rows = [row for row in hourly_rows if row["engine"] == engine and row["status"] == "ok"]
        six_minute_engine_rows = [row for row in six_minute_rows if row["engine"] == engine and row["status"] == "ok"]
        lines.append(
            "| {engine} | {hourly_rmse} | {six_rmse} | {hourly_p95} | {six_p95} |".format(
                engine=engine,
                hourly_rmse=_format_optional(_average_numeric(hourly_engine_rows, "rmse")),
                six_rmse=_format_optional(_average_numeric(six_minute_engine_rows, "rmse")),
                hourly_p95=_format_optional(_average_numeric(hourly_engine_rows, "p95_time_error_minutes")),
                six_p95=_format_optional(_average_numeric(six_minute_engine_rows, "p95_time_error_minutes")),
            )
        )

    lines.extend(
        [
            "",
            "## Station Deltas",
            "",
            "| Station | Engine | RMSE delta (6-minute - hourly) | Max extrema delta (min) | p95 extrema delta (min) |",
            "|--------|--------|-------------------------------|--------------------------|--------------------------|",
        ]
    )
    for station_id, engine in shared_keys:
        hourly_row = hourly_by_key[(station_id, engine)]
        six_row = six_minute_by_key[(station_id, engine)]
        lines.append(
            "| {station_name} ({station_id}) | {engine} | {rmse_delta} | {max_delta} | {p95_delta} |".format(
                station_name=hourly_row["station_name"],
                station_id=station_id,
                engine=engine,
                rmse_delta=_format_optional(_delta(six_row.get("rmse"), hourly_row.get("rmse"))),
                max_delta=_format_optional(_delta(six_row.get("max_time_error_minutes"), hourly_row.get("max_time_error_minutes"))),
                p95_delta=_format_optional(_delta(six_row.get("p95_time_error_minutes"), hourly_row.get("p95_time_error_minutes"))),
            )
        )

    highest_p95_row = None
    ok_six_rows = [row for row in six_minute_rows if row["status"] == "ok" and row.get("p95_time_error_minutes") not in ("", None)]
    if ok_six_rows:
        highest_p95_row = max(ok_six_rows, key=lambda row: float(row["p95_time_error_minutes"]))

    lines.extend(["", "## Findings", ""])
    if highest_p95_row is not None:
        lines.append(
            "- Highest 6-minute p95 extrema timing error: "
            f"{highest_p95_row['station_name']} ({highest_p95_row['engine']}) at {highest_p95_row['p95_time_error_minutes']} min."
        )
    hourly_p95 = _average_numeric([row for row in hourly_rows if row["status"] == "ok"], "p95_time_error_minutes")
    six_p95 = _average_numeric([row for row in six_minute_rows if row["status"] == "ok"], "p95_time_error_minutes")
    if hourly_p95 is not None and six_p95 is not None:
        delta = six_p95 - hourly_p95
        if delta < 0:
            lines.append(
                f"- Average p95 extrema timing error improves by {-delta:.4f} minutes when moving from hourly to 6-minute snapshots."
            )
        else:
            lines.append(
                f"- Average p95 extrema timing error increases by {delta:.4f} minutes when moving from hourly to 6-minute snapshots."
            )
    return "\n".join(lines) + "\n"


def write_resolution_comparison(hourly_summary_path, six_minute_summary_path, output_path):
    report = render_resolution_comparison(
        load_benchmark_summary(hourly_summary_path),
        load_benchmark_summary(six_minute_summary_path),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return str(destination)


def _average_numeric(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not values:
        return None
    return sum(values) / len(values)


def _delta(left, right):
    if left in ("", None) or right in ("", None):
        return None
    return float(left) - float(right)


def _format_optional(value):
    if value is None:
        return ""
    return f"{value:.4f}"
