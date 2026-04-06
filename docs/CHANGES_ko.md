# 변경 이력

## Unreleased
- 체크인된 다관측소 NOAA 스냅샷을 사용하는 repo-only benchmark harness 추가
- Markdown + CSV/JSON benchmark artifact 생성 기능 추가
- 동일 관측소 집합에 대한 `hourly` 및 `6-minute` NOAA benchmark 프로필 추가
- benchmark artifact에 `p95_time_error_minutes` 극값 지표 추가
- benchmark 흐름에 대한 회귀, 통합, 실패 경로 테스트 추가
- `UTide==0.3.1` 비교를 위한 선택 의존성 그룹 `benchmark` 추가
- `hourly`와 `6-minute` 결과를 비교하는 benchmark comparison report 추가

## 2026-04-06 - Release 0.8.1
- Python >= 3.10, < 3.14 지원
- 최소 의존성 하한을 NumPy >= 2.2.6, SciPy >= 1.15.3으로 조정

## 2025-07-17 - Release 0.8.0-py3
- Updated to Numpy 2.0

## 2022-06-10 - Release 0.0.4-py3
- Fixed Python 3.10.x

## 2013-11-24 - Release 0.0.3
- Fixed highs() and lows()
- Made at() use partition method rather than recurse.
- Documented Tide

## 2013-11-12 - Initial release 0.0.1
