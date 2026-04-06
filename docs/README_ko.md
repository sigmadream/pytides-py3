# pytides-py3

> pytides-py3는 조석 분석 및 예측을 위한 Python 라이브러리입니다. 기존 pytides를 현대적인 Python 및 NumPy/SciPy 환경에 맞게 갱신한 포크입니다.

## 설치

```bash
pip install pytides-py3
```

또는 `uv`를 사용하면 다음과 같습니다.

```bash
uv add pytides-py3
```

## 주요 특징

- 조석 분석 및 예측: 과거 조석 데이터를 기반으로 특정 위치의 조석 행동을 추정
- 조화분조법(Harmonic Constituents) 사용: P. Schureman의 Special Publication 98에 제시된 방법론 적용
- NOAA 조화분조 지원: NOAA에서 발표한 진폭과 위상을 직접 사용 가능
- SciPy 최적화: `scipy.optimize.least_squares`를 사용한 진폭과 위상 피팅

## 기술적 요구사항

- Python >= 3.10, < 3.15
- NumPy >= 2.2.6
- SciPy >= 1.15.3

## 주요 기능

- 조석 모델링: `Tide` 클래스를 통한 조석 예측
- 고조/저조 예측: `highs()`, `lows()` 메서드로 고조와 저조 시점 예측
- 조석 분류: form number를 통한 조석 유형 분류 (반일주조, 혼합조, 일주조)
- 시간대 처리: UTC datetime 형식 사용 (서머타임 조정 없음)

## 참고 문헌

- Original pytides: https://github.com/sam-cox/pytides
- Schureman, P. (1958). __Manual of Harmonic Analysis and Prediction of Tides__. U.S. Coast and Geodetic Survey, Special Publication No. 98.
  - https://tidesandcurrents.noaa.gov/publications/SpecialPubNo98.pdf
- Meeus, J. (1991). __Astronomical Algorithms__. Willmann-Bell.

## 개발

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv build
```

`uv` 기반 Python 버전 매트릭스 테스트:

```bash
bash scripts/test-python-matrix.sh
```

지원 범위만 또는 실험 범위만 따로 실행할 수도 있습니다.

```bash
bash scripts/test-python-matrix.sh --supported-only
bash scripts/test-python-matrix.sh --experimental-only
```

체크인된 NOAA 스냅샷으로 benchmark harness를 실행하려면 다음 명령을 사용합니다.

```bash
uv run python scripts/run_noaa_benchmark.py --output-dir ./benchmark_artifacts
```

극값(extrema) 비교를 더 촘촘하게 보고 싶다면 `6-minute` 프로필을 사용할 수 있습니다.

```bash
uv run python scripts/run_noaa_benchmark.py --dataset 6-minute --output-dir ./benchmark_artifacts_6min
```

시간 해상도별 차이를 한 번에 보는 비교 리포트도 생성할 수 있습니다.

```bash
uv run python scripts/compare_benchmark_artifacts.py
```

선택 의존성인 `UTide` 비교를 켜려면 다음 명령을 먼저 실행합니다.

```bash
uv sync --extra benchmark
```

현재 체크인된 benchmark 결과 해석은 다음과 같습니다.

- `hourly`와 `6-minute` 모두 평균 RMSE 기준으로는 `pytides-py3`와 `UTide`가 사실상 동률입니다.
- `6-minute` 프로필은 평균 RMSE를 거의 바꾸지 않으면서 평균 `p95 extrema timing error`를 약 `67분`에서 약 `39분` 수준으로 낮춥니다.
- 생성된 비교 리포트는 `benchmark_artifacts/benchmark-resolution-comparison.md`에 저장됩니다.

---

## Original README

Pytides is small Python package for the analysis and prediction of tides. Pytides can be used to extrapolate the tidal behaviour at a given location from its previous behaviour. The method used is that of harmonic constituents, in particular as presented by P. Schureman in Special Publication 98. The fitting of amplitudes and phases is handled by Scipy's leastsq minimisation function. Pytides currently supports the constituents used by NOAA, with plans to add more constituent sets. It is therefore possible to use the amplitudes and phases published by NOAA directly, without the need to perform the analysis again (although there may be slight discrepancies for some constituents). 

It is recommended that all interactions with pytides which require times to be specified are in the format of naive UTC datetime instances. In particular, note that pytides makes no adjustment for summertime or any other civil variations within timezones. 

For more information, please see http://github.com/sam-cox/pytides and https://github.com/sigmadream/pytides
