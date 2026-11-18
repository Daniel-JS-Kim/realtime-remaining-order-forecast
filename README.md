# 라이브 커머스 남은 주문 예측 — 분석 코드

박사학위논문 「실시간 판매 의사결정을 위한 라이브 커머스 잔여 주문량 예측: 베이지안 상태공간 모형을 활용한 방송 중 분 단위 예측」(김재식, 서강대학교)의 본 연구 모형 코드이다. 방송이 진행되는 매 분, 방송이 끝날 때까지 남은 주문 총량을 다시 예측한다.

논문의 원자료는 공개할 수 없으므로(아래 「자료 안내」), 같은 컬럼 구조의 가상 자료를 만들어 코드 전체를 실행해 볼 수 있게 하였다. 같은 코드를 실제 자료에 실행하면 논문의 수치가 그대로 재현된다(아래 「실제 자료에서의 재현 확인」).

## 파일 구성

| 파일 | 내용 |
|---|---|
| `make_example_data.py` | 실제 자료와 같은 컬럼 구조의 가상 자료를 만든다 (`example_data/broadcast_minutes.csv`) |
| `run_model.py` | 참조 곡선, 방송 전 예상 배율 μ\*, 매 분 갱신과 남은 주문 예측, 모수 조율, 가중 평균 nRMSE 평가 |
| `LICENSE` | MIT License |
| `requirements.txt` | 필요한 패키지 (numpy, pandas) |

## 실행 방법

```bash
pip install -r requirements.txt
python make_example_data.py     # 가상 자료 생성 (1~2초)
python run_model.py             # 모형 실행 (수 초)
```

같은 컬럼 구조의 다른 자료가 있으면 경로를 넘겨 실행한다.

```bash
python run_model.py <자료 경로.csv 또는 .pkl>
```

## 모형

방송 $i$ 의 $t$ 분 주문을 $N_t$, 같은 상품 카테고리·편성 조건의 과거 방송으로 만든 분당 평균 주문(참조 곡선)을 $Z_t$ 라 할 때, 참조 대비 판매 속도인 배율 $\theta_t$ 를 매 분 갱신한다.

| 단계 | 식 | 코드 위치 | 논문 위치 |
|---|---|---|---|
| 관측 | $N_t \mid \theta_t \sim \text{Poisson}(\theta_t Z_t)$ | — | 3.1.2 |
| 참조 곡선 | 상품 계층 5단계마다 해당 방송 이전 날짜의 최근 30회 방송의 분당 평균(주말·시간대가 같은 방송 우선), 다섯 계층의 평균 | `run_model.py` 2단계 | 3.2.1 |
| 방송 전 예상 배율 | $\hat\mu$: 같은 카테고리 과거 방송의 실제 배율 $Y/\sum Z$ 의 최근 30회 중앙값 <br> $\mu^* = \exp(c_\ell + b_\ell \log\hat\mu)$: 학습 구간에서 계층별로 적합한 축소 회귀 | 3단계 | 3.2.2 |
| 매 분 갱신 | $\alpha_t = \gamma\alpha_{t-1} + \mu^* b_0 + N_t$ <br> $\beta_t = \gamma\beta_{t-1} + b_0 + Z_t$, $\quad\hat\theta_t = \alpha_t/\beta_t$ | `run_and_forecast()` | 3.3.1 |
| $k$ 분 앞 배율 | $w_k\hat\theta_t + (1-w_k)\mu^*$, $\quad w_k = \dfrac{\gamma^k\beta_t}{\gamma^k\beta_t + b_0(1-\gamma^k)/(1-\gamma)}$ | `run_and_forecast()` | 3.3.2 |
| 남은 주문 | $\hat R(t) = \mu^*\sum_{s>t} Z_s + (\hat\theta_t - \mu^*)\sum_{k} w_k Z_{t+k}$ | `run_and_forecast()` | 3.3.2 |
| 평가 지표 | 방송별 $\text{nRMSE}_i = \sqrt{\tfrac{1}{T_i}\sum_t (\hat R_i(t) - R_i(t))^2}\,/\,Y_i$ <br> 가중 평균 nRMSE: 방송 총주문 $Y_i$ 를 가중치로 한 평균 | `score()` | 4.2.2 |

- 매 분 갱신에서 $\mu^* b_0$ 와 $b_0$ 는 배율이 $\mu^*$ 인 가상의 주문을 노출 $b_0$ 만큼 관측한 것으로 보고 더하는 값이다. 이 때문에 배율 추정이 $\mu^*$ 쪽으로 조금씩 되돌아가며(평균회귀), 사후분포는 감마분포로 유지된다.
- $w_k$ 는 지금까지의 관측으로 얻은 배율과 방송 전 예상 배율 사이의 가중치이다. 먼 분일수록 작아져 예측이 $\mu^*$ 쪽으로 옮겨 간다.
- 모수 $(\gamma, b_0)$ 는 학습 구간에서 $\gamma$ 12개 값 × $b_0$ 9개 값의 격자 가운데 가중 평균 nRMSE가 가장 작은 조합으로 정한다. $\mu^*$ 의 축소 회귀도 학습 구간에서 적합하며, 참조 곡선과 $\hat\mu$ 은 평가 대상 방송의 방송일 이전 자료만으로 만든다.

## 실제 자료에서의 재현 확인

`run_model.py` 를 수정 없이 실제 자료에 실행한 결과이다. 방송 13,079건 가운데 2025년 7~11월 평가 구간(논문의 검증·시험 구간) 2,904건의 집계값이며, 논문 <표 18>의 값과 소수 넷째 자리까지 같다. 학습 구간 조율로 고른 모수도 논문과 같은 $(\gamma, b_0) = (0.80, 0.10)$ 이다.

| 예측 | 가중 평균 nRMSE | 방송별 nRMSE 중앙값 |
|---|---|---|
| 참조 곡선만 ($\theta = 1$) | 0.2473 | 0.2218 |
| 방송 전 예측 ($\mu^* \sum Z$) | 0.2243 | 0.1803 |
| 매 분 갱신, 목표 1 | 0.1918 | 0.1735 |
| **본 연구 (매 분 갱신, 목표 $\mu^*$)** | **0.1782** | **0.1551** |

## 가상 자료에서의 실행 결과

가상 자료는 실제 자료와 분포가 다르므로 수치는 위 표와 다르다. 오차가 참조 곡선만, 방송 전 예측, 매 분 갱신(목표 1), 본 연구의 순서로 낮아지는 점은 실제 자료와 같다.

| 예측 | 가중 평균 nRMSE | 방송별 nRMSE 중앙값 |
|---|---|---|
| 참조 곡선만 ($\theta = 1$) | 0.5090 | 0.5850 |
| 방송 전 예측 ($\mu^* \sum Z$) | 0.3796 | 0.3442 |
| 매 분 갱신, 목표 1 | 0.3044 | 0.3298 |
| **본 연구 (매 분 갱신, 목표 $\mu^*$)** | **0.2758** | **0.2634** |

(2025년 7~11월 평가 구간 612건, 학습 구간 조율 모수 $(\gamma, b_0) = (0.97, 0.25)$)

## 자료 안내

- 논문의 자료는 국내 TV 홈쇼핑 A사의 사용 허락을 받아 분석하였다. 원자료는 공개·재배포 금지 조항에 따라 공개하지 않는다.
- `run_model.py` 가 읽는 자료는 방송 × 분 단위의 긴 형식 표이며, 다음 컬럼이 필요하다. 다른 자료에 적용하려면 이 구조로 변환하여 입력한다.

| 컬럼 | 뜻 |
|---|---|
| `KEY_ID` | 방송 식별자 |
| `SEQ` | 방송 시작부터의 분 (0부터) |
| `주문수량` | 그 분의 주문 수량 |
| `기준년월일` | 방송 날짜 (`YYYYMMDD` 문자열) |
| `분석대상` | 분석에 포함할 방송 여부 (총주문이 0인 방송은 제외) |
| `아이템코드`, `세분류카테고리`, `소분류카테고리`, `중분류카테고리`, `대분류카테고리` | 상품 계층 5단계 (좁은 단계부터) |
| `방송주말구분`, `방송시작시간구분` | 참조 곡선의 편성 조건 |

## 이 저장소에 없는 것

원자료의 전처리, 예측구간, 비교 모형(전통 통계 모형 등), 조건별 분석, 의사결정 시뮬레이션, 그림 코드는 원자료의 형식과 묶여 있어 넣지 않았다. 이 저장소는 본 연구 모형의 계산 논리를 확인하는 데 필요한 부분으로 구성하였다.

## 라이선스

MIT License (`LICENSE`)

---

## English summary

Code for the proposed model in the doctoral dissertation *Forecasting Remaining Orders in Live Commerce for Real-Time Sales Decisions: Minute-by-Minute Prediction with a Bayesian State-Space Model* (Jaesik Kim, Sogang University). At every minute of a live broadcast, the model updates a Poisson-Gamma state-space estimate of the selling rate relative to a reference curve, with conjugate mean reversion toward a broadcast-specific pre-broadcast multiplier μ\*, and forecasts the total remaining orders until the end of the broadcast. The original data were used with permission from a Korean TV home shopping company and cannot be shared; `make_example_data.py` generates synthetic data in the same format. Run `python make_example_data.py` and then `python run_model.py`. On the original data the same script reproduces the dissertation's results (weighted mean nRMSE 0.1782 on 2,904 evaluation broadcasts from July to November 2025).
