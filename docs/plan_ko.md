# pytides-py3 프로젝트 업데이트 계획

## 목표

1. 최신 Python 버전에서 사용할 수 있도록 업데이트
2. NumPy 2.x 및 최신 SciPy 버전 지원
3. UV 기반 패키지 빌드 및 배포

## 원칙

1. 타입힌팅은 Python에 표준으로 등장하기 전까지 사용하지 않음

## 현재 상태

- 버전: 0.8.1
- Python >= 3.10, < 3.14 지원
- NumPy >= 2.2.6, SciPy >= 1.15.3 지원
- UV 기반 빌드/배포 체계 (`pyproject.toml` + setuptools)
- 체크인된 NOAA 스냅샷 기반 benchmark harness 추가
- `hourly` 및 `6-minute` benchmark 프로필 추가
- `UTide==0.3.1` 비교 경로 추가
- `p95 extrema timing error`를 포함한 benchmark artifact 추가
- 테스트 134개 통과

### 현재 의존성

- `numpy>=2.2.6`
- `scipy>=1.15.3`

---

## ~~1단계: 의존성 업데이트~~

### 1.1 NumPy 업데이트
- [x] `numpy>=1.8` → `numpy>=2.2.6`
- [x] `np.divide` → `/` 연산자로 변경
- [x] NumPy 2.x API 호환성 검증 완료

### 1.2 SciPy 업데이트
- [x] `scipy>=0.11` → `scipy>=1.15.3`
- [x] `scipy.optimize.leastsq` → `scipy.optimize.least_squares` 전환

## ~~2단계: Python 호환성 개선~~
- [x] `python_requires='>=3.10,<3.14'`
- [x] Python 3.10, 3.11, 3.12, 3.13 지원
- [x] `collections.abc` 사용
- [x] f-string 사용

## ~~3단계: 패키지 빌드 체계 전환~~

- [x] `pyproject.toml`로 모든 메타데이터 이전 (빌드 백엔드: setuptools)
- [x] `setup.py`, `setup.cfg`, `MANIFEST.in` 삭제
- [x] `pytidespy3/__init__.py` 생성 (`__version__`, 주요 임포트)
- [x] `tests/__init__.py` 생성
- [x] `uv build` → sdist + wheel 생성 확인
- [x] `CLAUDE.md` 생성 (Claude Code 가이드)

## ~~4단계: 로버스트성 개선~~

- [x] `Tide.decompose()` — NaN/inf 자동 제거 (heights + 대응 시간 쌍 제거)
- [x] `Tide.decompose()` — `weights` 파라미터 추가 (가중 최소제곱)
- [x] `Tide.decompose()` — `loss` 파라미터 추가 (`'linear'`, `'huber'`, `'soft_l1'`, `'cauchy'`, `'arctan'`)
- [x] `Tide.at()` — 빈 배열 입력 검증
- [x] 로버스트성 테스트 6개 추가 (NaN, inf, 전체 NaN, weights, huber loss, 빈 배열)

## ~~6단계: PyPI 배포~~

- [x] 버전 0.8.1
- [x] `CHANGES`, `CHANGES.md`, `plan.md`, `README.md`, `docs/` 문서 업데이트
- [x] `uv build` → sdist + wheel 생성 확인

---

## 5단계: 학술 검증 (진행 중)

NOAA 실측 데이터를 활용한 정확도 검증

### 5.1 다관측소 장기 검증
- [x] 반일주/혼합조를 대표하는 3개 NOAA 관측소 스냅샷 수집
- [x] 약 3개월 시계열로 RMSE·MAE·상관·극값 오차 비교 harness 작성
- [x] San Francisco 1개소(1주일) 회귀 검증을 다관측소 benchmark로 확장
- [x] `pytides-py3`와 `UTide`를 같은 데이터와 같은 split으로 비교
- [x] Markdown + CSV + JSON artifact 생성
- [x] `hourly`와 `6-minute` 해상도 비교 리포트 생성

### 5.2 계절 변동 검증 (TODO #2)
- [ ] 동일 관측소의 계절별 구간(겨울/여름) 정확도 비교
- [ ] 기존 harness를 재사용해 계절별 benchmark window 추가

### 5.3 NOAA 공식 조화상수 비교 (TODO #3)
- [ ] NOAA CO-OPS에서 관측소별 공식 amplitude/phase 수집
- [ ] `Tide.decompose` 결과와 성분별(M2, S2, K1, O1 등) 비교 테스트 작성
- [ ] 허용 오차 기준 설정

### 5.4 benchmark 배포 자동화 (TODO #4)
- [ ] GitHub Actions에서 benchmark artifact를 생성하고 업로드
- [ ] 수동 실행 결과와 CI 결과의 일관성 검증

---

## 리스크

1. [x] 기존 사용자 호환성 유지 (검증 완료)
2. [x] NumPy/SciPy API 변경으로 인한 기능 손실 (검증 완료)
3. [ ] 성능 변화 모니터링 필요
4. [ ] 학술 검증을 위한 NOAA 데이터 확보 필요
