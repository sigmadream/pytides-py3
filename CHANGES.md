# Changelog

## 2026-04-06 - Release 0.8.1

- Python >= 3.10, < 3.14 support
- Lowered dependency floors to NumPy >= 2.2.6 and SciPy >= 1.15.3

## 2026-02-08 - Release 0.8.0

- Python >= 3.11, < 3.14 support
- NumPy >= 2.3.1, SciPy >= 1.16.0
- Migrated from `scipy.optimize.leastsq` to `scipy.optimize.least_squares`
- UV-based build system (`pyproject.toml` + setuptools)
- Removed `setup.py`, `setup.cfg`, `MANIFEST.in`
- Fixed s/h variable swap in `astro.py` (Schureman convention: s=lunar, h=solar)
- Added constituent speed validation tests (`M2`, `S2`, `K1`, `O1`, `N2`)
- `Tide.decompose()`: auto-remove `NaN`/`inf`, `weights` parameter, `loss` parameter
- `Tide.at()`: empty array validation
- 120 tests (unit, integration, NOAA cross-check, robustness)

## 2025-07-17 - Release 0.8.0-py3

- Updated to Numpy 2.0

## 2022-06-10 - Release 0.0.4-py3

- Fixed Python 3.10.x

## 2013-11-24 - Release 0.0.3

- Fixed `highs()` and `lows()`
- Made `at()` use partition method rather than recurse
- Documented `Tide`

## 2013-11-12 - Initial release 0.0.1
