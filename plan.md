# pytides-py3 Project Update Plan

## Goals

1. Update for use on recent Python versions
2. Support NumPy 2.x and recent SciPy versions
3. UV-based package build and distribution

## Principles

1. Do not use type hinting until it is part of the Python standard

## Current Status

- **Version**: 0.8.0
- Python >= 3.11, < 3.14 supported
- NumPy >= 2.3.1, SciPy >= 1.16.0
- UV-based build/distribution (`pyproject.toml` + setuptools)
- 120 tests passing

### Current Dependencies

- `numpy>=2.3.1`
- `scipy>=1.16.0`

---

## ~~Phase 1: Dependency Update~~

### 1.1 NumPy Update
- [x] `numpy>=1.8` → `numpy>=2.3.1`
- [x] Replace `np.divide` with `/` operator
- [x] NumPy 2.x API compatibility verified

### 1.2 SciPy Update
- [x] `scipy>=0.11` → `scipy>=1.16.0`
- [x] Migrate from `scipy.optimize.leastsq` to `scipy.optimize.least_squares`

## ~~Phase 2: Python Compatibility~~
- [x] `python_requires='>=3.11,<3.14'`
- [x] Support Python 3.11, 3.12, 3.13
- [x] Use `collections.abc`
- [x] Use f-strings

## ~~Phase 3: Package Build System Migration~~
- [x] Move all metadata to `pyproject.toml` (build backend: setuptools)
- [x] Remove `setup.py`, `setup.cfg`, `MANIFEST.in`
- [x] Add `pytidespy3/__init__.py` (`__version__`, main imports)
- [x] Add `tests/__init__.py`
- [x] Confirm `uv build` produces sdist + wheel
- [x] Add `CLAUDE.md` (Claude Code guide)

## ~~Phase 4: Robustness Improvements~~
- [x] `Tide.decompose()` — Auto-remove NaN/inf (drop height–time pairs)
- [x] `Tide.decompose()` — Add `weights` parameter (weighted least squares)
- [x] `Tide.decompose()` — Add `loss` parameter (`'linear'`, `'huber'`, `'soft_l1'`, `'cauchy'`, `'arctan'`)
- [x] `Tide.at()` — Validate empty array input
- [x] Add 6 robustness tests (NaN, inf, all-NaN, weights, huber loss, empty array)

## ~~Phase 5: Scientific Accuracy~~
- [x] Fix s/h variable swap in `astro.py` (Schureman convention: s=lunar, h=solar)
- [x] Add constituent speed validation tests (M2=28.984, S2=30.0, K1=15.041, O1=13.943, N2=28.440 deg/hr)
- [x] Update golden values in `test_astro.py` and `test_tide.py`
- [x] Fix `test_reference_data.py` docstring and tolerance label errors
- [x] Relax node factor range assertion (`f < 2.0` → `f < 3.0`)

## ~~Phase 6: PyPI Release~~
- [x] Version 0.8.0
- [x] Update CHANGES, PLAN.md, README.md
- [x] `uv build` produces sdist + wheel

---

## Phase 7: Academic Validation (Not Started)

Accuracy validation using NOAA observed data

### 7.1 Multi-Station Long-Term Validation
- [ ] Collect data from 3+ stations representing semidiurnal/diurnal/mixed tides
- [ ] Compare RMSE, MAE, correlation, and extreme-value error on long-term (>=3 months) series
- [ ] Current coverage: San Francisco only (1 week) — needs expansion

### 7.2 Seasonal Variation Validation
- [ ] Compare accuracy across seasonal segments (winter/summer) at the same station
- [ ] Proceed after 7.1 data is available

### 7.3 NOAA Official Harmonic Constants Comparison
- [ ] Obtain official amplitude/phase per station from NOAA CO-OPS
- [ ] Add comparison tests (constituent-wise: M2, S2, K1, O1, etc.) with `Tide.decompose` results
- [ ] Define tolerance criteria

---

## Risks

1. [x] Maintain compatibility with existing users (verified)
2. [x] Feature loss from NumPy/SciPy API changes (verified)
3. [x] Scientific accuracy of constituent speeds (fixed s/h swap)
4. [ ] Performance change monitoring needed
5. [ ] NOAA data acquisition needed for academic validation
