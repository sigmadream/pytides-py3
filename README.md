# pytides-py3

> pytides-py3는 조석 분석 및 예측을 위한 Python 라이브러리입니다. 이는 기존 pytides의 개선된 버전으로, Python 3.12.x에서 작동하도록 업데이트되었습니다.

## 실행

```bash
pip install -e .
python -m unittest discover -s tests
```

## 주요 특징:

- 조석 분석 및 예측: 과거 조석 데이터를 기반으로 특정 위치의 조석 행동을 추정
- 조화분조법(Harmonic Constituents) 사용: P. Schureman의 Special Publication 98에 제시된 방법론 적용
- NOAA 조화분조 지원: NOAA에서 발표한 진폭과 위상을 직접 사용 가능
- Scipy 최적화: leastsq 최소화 함수를 사용한 진폭과 위상 피팅

## 기술적 요구사항:

- Python >= 3.11.x
- NumPy >= 1.8
- SciPy >= 0.11

## 주요 기능:

- 조석 모델링: `Tide` 클래스를 통한 조석 예측
- 고조/저조 예측: `highs()`, `lows()` 메서드로 고조와 저조 시점 예측
- 조석 분류: form number를 통한 조석 유형 분류 (반일주조, 혼합조, 일주조)
- 시간대 처리: UTC datetime 형식 사용 (서머타임 조정 없음)

## 참고 문헌

---

## Original README

Pytides is small Python package for the analysis and prediction of tides. Pytides can be used to extrapolate the tidal behaviour at a given location from its previous behaviour. The method used is that of harmonic constituents, in particular as presented by P. Schureman in Special Publication 98. The fitting of amplitudes and phases is handled by Scipy's leastsq minimisation function. Pytides currently supports the constituents used by NOAA, with plans to add more constituent sets. It is therefore possible to use the amplitudes and phases published by NOAA directly, without the need to perform the analysis again (although there may be slight discrepancies for some constituents). 

It is recommended that all interactions with pytides which require times to be specified are in the format of naive UTC datetime instances. In particular, note that pytides makes no adjustment for summertime or any other civil variations within timezones. 

For more information, please see http://github.com/sam-cox/pytides and https://github.com/sigmadream/pytides
