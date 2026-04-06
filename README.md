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

## References

- Original pytides: https://github.com/sam-cox/pytides
- Schureman, P. (1958). __Manual of Harmonic Analysis and Prediction of Tides__. U.S. Coast and Geodetic Survey, Special Publication No. 98.
    - [Link1](https://tidesandcurrents.noaa.gov/publications/SpecialPubNo98.pdf)
- Meeus, J. (1991). __Astronomical Algorithms__. Willmann-Bell.

## License

MIT
