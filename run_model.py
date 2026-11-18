"""
라이브 커머스 남은 주문 예측 — 본 연구 모형 실행

방송이 진행되는 매 분, 방송이 끝날 때까지 남은 주문 총량을 다시 예측한다.

[모형 요약]
  관측       N_t | θ_t ~ Poisson(θ_t · Z_t)          Z_t: 참조 곡선(과거 비슷한 방송의 분당 평균 주문)
  매 분 갱신  α_t = γ·α_{t-1} + μ*·b₀ + N_t          θ_t: 참조 대비 배율, 사후 평균 θ̂_t = α_t / β_t
             β_t = γ·β_{t-1} + b₀    + Z_t          μ*: 방송 전 예상 배율, b₀: 가상 주문의 노출
  k분 앞 배율 θ̃_{t+k|t} = w_k·θ̂_t + (1 − w_k)·μ*     (t분 이후 관측 없이 예측 단계만 k번 반복한 배율)
             w_k = γ^k·β_t / (γ^k·β_t + b₀·(1 − γ^k)/(1 − γ))
  남은 주문   R̂(t) = μ*·Σ_{s>t} Z_s + (θ̂_t − μ*)·Σ_k w_k·Z_{t+k}
  평가 지표   방송별 nRMSE = sqrt(평균_t ((R̂(t) − R(t)) / Y)²),  가중 평균 nRMSE = 방송 총주문 Y 가중 평균

[단계]
  1. 자료 로드 → 방송 × 분 주문 행렬
  2. 참조 곡선 Z (상품 계층 5단계 × 편성 조건, 해당 방송 이전 날짜의 방송만 사용)
  3. 방송 전 예상 배율 μ* (같은 카테고리 과거 방송의 실제 배율 중앙값 μ̂ 을 학습 구간 회귀로 축소)
  4. 모수 (γ, b₀) 조율 — 학습 구간에서 가중 평균 nRMSE 최소인 격자점
  5. 평가 구간(검증·시험) 평가 — 단계별 비교
  6. 예시 방송 한 건의 분별 예측

실행:  python run_model.py                         (예시 자료, make_example_data.py 먼저 실행)
       python run_model.py <자료 경로.csv|.pkl>      (같은 컬럼 구조의 다른 자료)
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── 0. 설정 ──────────────────────────────────────────────────────
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else 'example_data/broadcast_minutes.csv'

TEST_START = '20250701'           # 이 날짜부터 평가 구간(논문의 검증 7~9월 + 시험 10~11월). 모든 모수·회귀는 그 이전 자료로 정한다
Z_FLOOR = 1e-3                    # 참조 곡선 하한 (0 나눗셈 방지)
MIN_REF = 5                       # 참조 곡선을 만들 최소 과거 방송 수
K_REF = 30                        # 참조 곡선·μ̂ 에 쓰는 최근 과거 방송 수
MAX_SEQ = 130                     # 분석할 최대 분
MIN_HIST = 3                      # μ̂ 추정에 필요한 최소 과거 방송 수
MU_CLIP = (0.15, 4.0)             # μ̂·μ* 범위
MIN_FIT = 200                     # 계층별로 따로 축소 회귀를 적합할 최소 표본 수
LEVELS = [('LV1', '아이템코드'), ('LV2', '세분류카테고리'), ('LV3', '소분류카테고리'),
          ('LV4', '중분류카테고리'), ('LV5', '대분류카테고리')]
REF_COND = ['방송주말구분', '방송시작시간구분']     # 참조 곡선의 편성 조건
GAMMA_GRID = [.30, .40, .50, .60, .70, .80, .85, .88, .91, .94, .97, .99]
B0_GRID = [.02, .05, .10, .25, .50, 1.0, 2.0, 4.0, 8.0]

# ── 1. 자료 로드 ─────────────────────────────────────────────────
df = pd.read_pickle(DATA_PATH) if DATA_PATH.endswith('.pkl') else pd.read_csv(DATA_PATH, dtype={'KEY_ID': str, '기준년월일': str})
df = df[df['분석대상'] & (df['SEQ'] <= MAX_SEQ)]

# 방송 × 분 주문 행렬. 방송이 끝난 뒤의 칸은 NaN
mat = df.pivot_table(index='KEY_ID', columns='SEQ', values='주문수량', aggfunc='sum')
meta = df.groupby('KEY_ID').agg(날짜=('기준년월일', 'first'),
                                **{c: (c, 'first') for _, c in LEVELS},
                                **{c: (c, 'first') for c in REF_COND}).loc[mat.index]
for c in REF_COND:
    meta[c] = meta[c].astype(str)

N = mat.to_numpy(float); obs = ~np.isnan(N); Nz = np.nan_to_num(N)
n_b, n_t = N.shape
cum = np.cumsum(Nz, 1)
Y = np.nanmax(np.where(obs, cum, np.nan), 1)          # 방송 총주문
R_true = Y[:, None] - cum                              # t분까지 본 뒤 실제 남은 주문
dates = meta['날짜'].to_numpy()
print(f"자료 로드 완료 | 방송 {n_b:,}건 · 최대 {n_t}분 | {datetime.now().strftime('%H:%M:%S')}")

# ── 2. 참조 곡선 Z ───────────────────────────────────────────────
# 같은 계층에서 해당 방송보다 이전 날짜의 최근 K_REF 개 방송의 분당 주문을 평균한다.
# 길이가 짧은 과거 방송은 남은 분을 자기 분당 평균으로 채운다.
own = np.nanmean(np.where(obs, Nz, np.nan), axis=1)
padded = np.where(obs, Nz, own[:, None])


def level_Z(keys):
    Zl = np.full((n_b, n_t), np.nan)
    for _, gi in meta.groupby(keys).indices.items():
        o = gi[np.argsort(dates[gi], kind='stable')]
        Cs = np.vstack([np.zeros((1, n_t)), np.cumsum(padded[o], axis=0)])
        hi = np.searchsorted(dates[o], dates[o], side='left')     # 같은 날 방송은 쓰지 않는다
        lo = np.maximum(hi - K_REF, 0); cnt = hi - lo
        ok = cnt >= MIN_REF
        if ok.any():
            Zl[o[ok]] = (Cs[hi[ok]] - Cs[lo[ok]]) / cnt[ok][:, None]
    return Zl


# 계층마다 편성 조건(주말 × 시간대)이 같은 방송으로 만들고, 표본이 모자라면 조건 없이 만든다.
# 다섯 계층 가운데 만들어진 것들을 평균한다.
S = np.zeros((n_b, n_t)); C = np.zeros((n_b, n_t))
for nm, col in LEVELS:
    Zu = level_Z([col]); Zk = level_Z([col] + REF_COND)
    Zl = np.where(~np.isnan(Zk[:, 0])[:, None], Zk, Zu)
    m = ~np.isnan(Zl); S[m] += Zl[m]; C[m] += 1
Z = np.maximum(np.where(C > 0, S / np.maximum(C, 1), np.nan), Z_FLOOR)
has_Z = ~np.isnan(Z[:, 0])
Zsum = np.where(obs, np.nan_to_num(Z), 0.0).sum(1)
ratio = np.where((Zsum > 0) & has_Z, Y / np.maximum(Zsum, 1e-9), np.nan)    # 실제 배율 = Y / ΣZ
print(f"참조 곡선 완료 | 참조가 있는 방송 {has_Z.sum():,}건 | {datetime.now().strftime('%H:%M:%S')}")

# ── 3. 방송 전 예상 배율 μ* ──────────────────────────────────────
# μ̂: 좁은 계층부터 과거 방송이 MIN_HIST 회 이상인 첫 계층의 최근 K_REF 회 실제 배율 중앙값


def hist_ratio(col):
    out = np.full(n_b, np.nan)
    for _, gi in meta.groupby(col).indices.items():
        o = gi[np.argsort(dates[gi], kind='stable')]
        r = ratio[o]
        hi = np.searchsorted(dates[o], dates[o], side='left')
        for j in range(len(o)):
            past = r[:hi[j]]
            past = past[~np.isnan(past)]
            if len(past) >= MIN_HIST:
                out[o[j]] = np.median(past[-K_REF:])
    return out


MU_RAW = np.full(n_b, np.nan); mu_lvl = np.array(['기본'] * n_b, dtype=object)
for nm, col in LEVELS:
    h = hist_ratio(col)
    need = np.isnan(MU_RAW) & ~np.isnan(h)
    MU_RAW[need] = h[need]; mu_lvl[need] = nm
MU_HAT = np.clip(np.where(np.isnan(MU_RAW), 1.0, MU_RAW), *MU_CLIP)

test = (dates >= TEST_START) & has_Z
train = (dates < TEST_START) & has_Z

# 축소 회귀: 학습 구간에서 log(실제 배율) = c + b·log(μ̂) 를 계층별로 적합해 μ* = exp(c + b·log μ̂).
# 표본이 MIN_FIT 에 못 미치는 계층은 전체 적합식을 쓴다
fit_ok = train & ~np.isnan(ratio) & (ratio > 0) & (MU_HAT > 0)
x_fit = np.log(MU_HAT[fit_ok]); y_fit = np.log(ratio[fit_ok]); lv_fit = mu_lvl[fit_ok]
b_all, a_all = np.polyfit(x_fit, y_fit, 1)
xlog = np.log(np.maximum(MU_HAT, 1e-9))
MU_STAR = np.exp(a_all + b_all * xlog)
slopes = {'전체': round(float(b_all), 3)}
for nm in ['LV1', 'LV2', 'LV3', 'LV4', 'LV5', '기본']:
    m_ = (lv_fit == nm)
    if m_.sum() >= MIN_FIT:
        b_, a_ = np.polyfit(x_fit[m_], y_fit[m_], 1)
        mm = (mu_lvl == nm); MU_STAR[mm] = np.exp(a_ + b_ * xlog[mm])
        slopes[nm] = round(float(b_), 3)
MU_STAR = np.clip(MU_STAR, *MU_CLIP)
print(f"μ* 완료 | 학습 {train.sum():,}건 · 평가 {test.sum():,}건 | 축소 기울기 {slopes} "
      f"| {datetime.now().strftime('%H:%M:%S')}")

# ── 4. 매 분 갱신·남은 주문 예측 함수와 모수 조율 ────────────────
# 격자 조율과 단계별 비교에서 같은 계산을 여러 번 부르므로 이 둘만 함수로 둔다


def run_and_forecast(Nr, Or, Zr, mu, gamma, b0):
    """방송별 배열을 받아 매 분 θ̂_t 를 갱신하고 그 시점의 남은 주문 예측 R̂(t) 를 돌려준다. 사전은 Gamma(μ*, 1)"""
    n = Nr.shape[0]
    Zc = np.where(Or, np.nan_to_num(Zr), 0.0)
    S1 = np.zeros((n, n_t))                       # S1[:, t] = Σ_{s>t} Z_s
    for t in range(n_t - 2, -1, -1):
        S1[:, t] = S1[:, t + 1] + Zc[:, t + 1]
    # 매 분 갱신
    a = mu.copy(); b = np.ones(n)
    th = np.full((n, n_t), np.nan); Bs = np.full((n, n_t), np.nan)
    for t in range(n_t):
        m = Or[:, t] & ~np.isnan(Zr[:, t])
        if not m.any():
            continue
        a[m] = gamma * a[m] + mu[m] * b0 + Nr[m, t]
        b[m] = gamma * b[m] + b0 + Zr[m, t]
        th[m, t] = a[m] / b[m]; Bs[m, t] = b[m]
    # 남은 각 분의 가중치 w_k 와 참조 곡선의 곱을 더한 값
    W = np.zeros((n, n_t))
    for t in range(n_t - 1):
        k = np.arange(1, n_t - t); gk = gamma ** k
        Gk = (1 - gk) / (1 - gamma)
        bt = np.nan_to_num(Bs[:, t], nan=1.0)
        w = gk[None, :] * bt[:, None] / (gk[None, :] * bt[:, None] + b0 * Gk[None, :] + 1e-12)
        W[:, t] = (w * Zc[:, t + 1:]).sum(1)
    return mu[:, None] * S1 + (th - mu[:, None]) * W


def score(R, Rr, Or, Yr):
    """방송별 nRMSE 와 총주문 가중 평균 nRMSE"""
    rel = np.where(Or & ~np.isnan(R), (R - Rr) / Yr[:, None], np.nan)
    per = np.sqrt(np.nanmean(rel ** 2, axis=1))
    ok = ~np.isnan(per)
    return np.average(per[ok], weights=Yr[ok]), per


L = dict(N=Nz[train], O=obs[train], Z=Z[train], R=R_true[train], Y=Y[train], mu=MU_STAR[train])
grid = []
for gm in GAMMA_GRID:
    for bb in B0_GRID:
        R = run_and_forecast(L['N'], L['O'], L['Z'], L['mu'], gm, bb)
        grid.append((score(R, L['R'], L['O'], L['Y'])[0], gm, bb))
best = min(grid)
GAMMA, B0 = best[1], best[2]
print(f"모수 조율 완료 | 격자 {len(grid)}점 · 학습 구간 최소 {best[0]:.4f} at γ = {GAMMA}, b₀ = {B0} "
      f"| {datetime.now().strftime('%H:%M:%S')}")

# ── 5. 평가 구간 평가 ────────────────────────────────────────────
T = dict(N=Nz[test], O=obs[test], Z=Z[test], R=R_true[test], Y=Y[test], mu=MU_STAR[test])
n = len(T['Y']); one = np.ones(n)
R_model = run_and_forecast(T['N'], T['O'], T['Z'], T['mu'], GAMMA, B0)
R_target1 = run_and_forecast(T['N'], T['O'], T['Z'], one, GAMMA, B0)
Zc_T = np.where(T['O'], np.nan_to_num(T['Z']), 0.0)
S1_T = np.flip(np.cumsum(np.flip(Zc_T, 1), 1), 1) - Zc_T        # Σ_{s>t} Z_s
STAGES = {
    '참조 곡선만 (θ = 1)': S1_T,
    '방송 전 예측 (μ*·ΣZ)': T['mu'][:, None] * S1_T,
    '매 분 갱신, 목표 1': R_target1,
    '본 연구 (매 분 갱신, 목표 μ*)': R_model,
}
print(f"\n평가 구간 {n:,}건 (기준일 {TEST_START} 이후), 모수 γ = {GAMMA}, b₀ = {B0}")
print(f"  {'예측':32s} {'가중 평균 nRMSE':>14s} {'중앙값':>8s}")
for nm, R in STAGES.items():
    pooled, per = score(R, T['R'], T['O'], T['Y'])
    print(f"  {nm:32s} {pooled:14.4f} {np.nanmedian(per):8.4f}")
print(f"평가 완료 | {datetime.now().strftime('%H:%M:%S')}")

# ── 6. 예시 방송 한 건 ───────────────────────────────────────────
# 평가 구간에서 총주문이 중앙값에 가장 가까운 방송
i = int(np.argmin(np.abs(T['Y'] - np.median(T['Y']))))
len_i = int(T['O'][i].sum())
print(f"\n예시 방송 {mat.index[test][i]} | 총주문 {T['Y'][i]:,.0f} · {len_i}분 · μ* {T['mu'][i]:.3f}")
print(f"  {'경과 분':>6s} {'누적 주문':>9s} {'실제 남은 주문':>13s} {'예측':>9s}")
for t in range(0, len_i, 10):
    print(f"  {t + 1:6d} {T['Y'][i] - T['R'][i, t]:9,.0f} {T['R'][i, t]:13,.0f} {R_model[i, t]:9,.0f}")
print(f"예시 출력 완료 | {datetime.now().strftime('%H:%M:%S')}")
