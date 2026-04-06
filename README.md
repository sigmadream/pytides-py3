# pytides-py3

> Tidal analysis and prediction library for Python 3.10+.
> An improved fork of the original [pytides](https://github.com/sam-cox/pytides), updated for NumPy 2.x and modern SciPy.

## Installation

```bash
pip install pytides-py3
```

Or with UV:

```bash
uv add pytides-py3
```

## Quick Start

```python
from datetime import datetime, timedelta
from pytidespy3 import Tide, constituent

# Build a model from known constituents
model = Tide(
    constituents=[constituent._M2, constituent._S2, constituent._K1],
    amplitudes=[1.0, 0.5, 0.3],
    phases=[0.0, 90.0, 180.0],
)

# Predict tidal heights
times = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(24)]
heights = model.at(times)

# Get high/low water times
highs = list(model.highs(datetime(2023, 1, 1), datetime(2023, 1, 8)))
lows = list(model.lows(datetime(2023, 1, 1), datetime(2023, 1, 8)))

# Decompose observed data into harmonic constituents
fitted = Tide.decompose(heights=observed_heights, t=observed_times)
```

## Main Features

- Tidal analysis and prediction via Schureman's harmonic constituent method
- 37 NOAA harmonic constituents (M2, S2, K1, O1, N2, etc.)
- Robust fitting: NaN/inf auto-removal, weighted least squares, robust loss functions (`huber`, `soft_l1`, `cauchy`, `arctan`)
- High/low water prediction with `highs()`, `lows()`, `extrema()`
- Tide classification: form number-based (semidiurnal, mixed, diurnal)

## Requirements

- Python >= 3.10, < 3.14
- NumPy >= 2.2.6
- SciPy >= 1.15.3

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv build
```

Python version matrix test with `uv`:

```bash
bash scripts/test-python-matrix.sh
```

Supported-only or experimental-only runs are also available:

```bash
bash scripts/test-python-matrix.sh --supported-only
bash scripts/test-python-matrix.sh --experimental-only
```

Benchmark harness with checked-in NOAA snapshots:

```bash
uv run python scripts/run_noaa_benchmark.py --output-dir ./benchmark_artifacts
```

Use the denser 6-minute NOAA profile when you want a stricter extrema benchmark:

```bash
uv run python scripts/run_noaa_benchmark.py --dataset 6-minute --output-dir ./benchmark_artifacts_6min
```

Compare the generated hourly and 6-minute artifacts in one Markdown report:

```bash
uv run python scripts/compare_benchmark_artifacts.py
```

Current checked-in benchmark readout:

- `hourly` and `6-minute` both read as a near-tie on average RMSE between `pytides-py3` and `UTide`.
- The `6-minute` profile keeps average RMSE effectively unchanged while improving average `p95 extrema timing error` from about `67.0 min` to about `39 min`.
- The generated comparison report lives at `benchmark_artifacts/benchmark-resolution-comparison.md`.

Install the optional comparison dependency:

```bash
uv sync --extra benchmark
```

Then run the full comparison:

```bash
uv run python scripts/run_noaa_benchmark.py --output-dir ./benchmark_artifacts
```

`UTide` comparison is optional. If `utide` is not installed, the runner records an explicit failure in the report instead of silently skipping it.

## References

- Original pytides: https://github.com/sam-cox/pytides
- Schureman, P. (1958). __Manual of Harmonic Analysis and Prediction of Tides__. U.S. Coast and Geodetic Survey, Special Publication No. 98.
    - [Link1](https://tidesandcurrents.noaa.gov/publications/SpecialPubNo98.pdf)
- Meeus, J. (1991). __Astronomical Algorithms__. Willmann-Bell.

## License

MIT
