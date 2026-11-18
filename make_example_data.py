"""
예시 자료 만들기 — 실제 방송 자료와 같은 형식의 가상 자료

원자료는 공개할 수 없어(README 참고) 같은 컬럼 구조의 가상 자료를 만든다.
값은 모두 아래 가정으로 난수 생성한 것이며 실제 방송·상품·카테고리와 관계없다.

[가정]
  - 2024-01-01 ~ 2025-11-30, 하루 4회 방송, 방송 길이 40~80분
  - 상품 계층 5단계(대분류 4 → 중분류 12 → 소분류 24 → 세분류 48 → 아이템 192)
  - 분당 주문 = 아이템 기본 수준 × 대분류별 진행 곡선 × 시간대·주말 효과
               × 방송별 배율(로그정규) × 같은 아이템 반복 방송 감쇠
               × 방송 중 흐름 변화(로그 랜덤워크) × 분별 변동(감마)
    을 평균으로 하는 포아송 난수

실행: python make_example_data.py
출력: example_data/broadcast_minutes.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

OUT_PATH = 'example_data/broadcast_minutes.csv'
SEED = 0
N_PER_DAY = 4
LENGTHS = [40, 50, 60, 60, 60, 70, 80]
SLOTS = {'새벽': 0.4, '아침': 0.9, '오전': 1.0, '점심': 1.1,
         '오후': 1.0, '저녁': 1.4, '야간': 1.3, '심야': 0.6}
WEEKEND_EFFECT = 1.15
DRIFT_SD = 0.10                   # 방송 중 흐름 변화의 분당 표준편차(로그 척도)

rng = np.random.default_rng(SEED)

# ── 1. 상품 계층 만들기 ──────────────────────────────────────────
items = []
for d in range(4):
    for m in range(3):
        for s in range(2):
            for f in range(2):
                for i in range(4):
                    items.append(dict(
                        대분류카테고리=f'D{d}',
                        중분류카테고리=f'D{d}M{m}',
                        소분류카테고리=f'D{d}M{m}S{s}',
                        세분류카테고리=f'D{d}M{m}S{s}F{f}',
                        아이템코드=f'D{d}M{m}S{s}F{f}I{i}'))
items = pd.DataFrame(items)
# 아이템 기본 수준(분당 주문)과 처음 방송할 수 있는 날(신상품이 기간 중간에 들어오도록)
items['기본수준'] = rng.lognormal(mean=2.3, sigma=0.8, size=len(items))
items['출시일순번'] = rng.integers(0, 600, size=len(items))
items.loc[rng.random(len(items)) < 0.5, '출시일순번'] = 0
print(f"상품 계층 완료 | 아이템 {len(items)}개 | {datetime.now().strftime('%H:%M:%S')}")

# ── 2. 대분류별 진행 곡선 (평균 1로 맞춤) ───────────────────────
shape_par = {f'D{d}': (rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.8), rng.uniform(0.0, 0.6))
             for d in range(4)}


def progress_curve(n_min, par):
    # 시작 직후 오르고 가운데 정점, 끝 무렵 마감 주문이 몰리는 모양
    p = np.linspace(0, 1, n_min)
    a, c, e = par
    f = 1 + a * np.exp(-((p - c) / 0.2) ** 2) + e * (p > 0.85) - 0.5 * np.exp(-p / 0.05)
    return f / f.mean()


print(f"진행 곡선 완료 | 대분류 {len(shape_par)}개 | {datetime.now().strftime('%H:%M:%S')}")

# ── 3. 편성 만들기 ───────────────────────────────────────────────
days = pd.date_range('2024-01-01', '2025-11-30', freq='D')
slot_names = list(SLOTS)
rows = []
air_count = {}
key = 0
for di, day in enumerate(days):
    weekend = '주말' if day.weekday() >= 5 else '평일'
    avail = items.index[items['출시일순번'] <= di].to_numpy()
    for slot in rng.choice(slot_names, size=N_PER_DAY, replace=False):
        it = items.loc[rng.choice(avail)]
        n_prev = air_count.get(it['아이템코드'], 0)
        air_count[it['아이템코드']] = n_prev + 1
        n_min = int(rng.choice(LENGTHS))
        level = (it['기본수준'] * SLOTS[slot] * (WEEKEND_EFFECT if weekend == '주말' else 1.0)
                 * rng.lognormal(0, 0.45) * 0.97 ** n_prev)
        # 방송 중 흐름 변화: 로그 척도 랜덤워크
        drift = np.zeros(n_min)
        for t in range(1, n_min):
            drift[t] = drift[t - 1] + rng.normal(0, DRIFT_SD)
        lam = (level * progress_curve(n_min, shape_par[it['대분류카테고리']]) * np.exp(drift)
               * rng.gamma(4.0, 1 / 4.0, size=n_min))
        orders = rng.poisson(lam)
        key += 1
        for seq in range(n_min):
            rows.append((f'B{key:05d}', seq, int(orders[seq]), day.strftime('%Y%m%d'),
                         it['아이템코드'], it['세분류카테고리'], it['소분류카테고리'],
                         it['중분류카테고리'], it['대분류카테고리'], weekend, slot))
df = pd.DataFrame(rows, columns=['KEY_ID', 'SEQ', '주문수량', '기준년월일', '아이템코드', '세분류카테고리',
                                 '소분류카테고리', '중분류카테고리', '대분류카테고리',
                                 '방송주말구분', '방송시작시간구분'])
print(f"편성 완료 | 방송 {key:,}건 · 분 {len(df):,}행 | {datetime.now().strftime('%H:%M:%S')}")

# ── 4. 분석 대상 표시와 저장 ─────────────────────────────────────
# 총주문이 0인 방송은 남은 주문 비율을 정의할 수 없어 제외한다(실제 자료와 같은 규칙)
total = df.groupby('KEY_ID')['주문수량'].transform('sum')
df['분석대상'] = total > 0
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f"저장 완료 | {OUT_PATH} | 분석 대상 방송 {df.loc[df['분석대상'], 'KEY_ID'].nunique():,}건 "
      f"| {datetime.now().strftime('%H:%M:%S')}")
