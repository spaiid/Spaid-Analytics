# scripts/generate_toy_features.py
import os
from pathlib import Path
import numpy as np
import polars as pl
from datetime import date, timedelta

# ------------------ Config ------------------
OUT_ROOT = Path(os.environ.get("SF_DATA_ROOT", "./_data")) / "features"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

H = int(os.environ.get("FWD_H", "20"))          # forward horizon to bake into file name + targets
N_DATES = int(os.environ.get("N_DATES", "260")) # ~1Y of trading days
N_ENTS  = int(os.environ.get("N_ENTS",  "50"))  # cross-sectional size (excl. benchmark)
SEED    = int(os.environ.get("SEED",    "7"))
USE_EXCESS = os.environ.get("TARGET", "ex_mkt").lower() in ("ex_mkt","both")

# Entities and sectors
rng = np.random.default_rng(SEED)
entities = [f"E{i:02d}" for i in range(1, N_ENTS+1)]
sectors  = [f"S{(i%5)+1}" for i in range(N_ENTS)]  # 5 sectors in rotation
benchmark = "SPY"

# Dates (business-ish days)
start = date(2020,1,2)
dates = [start + timedelta(days=i) for i in range(N_DATES)]
# Keep only weekdays
dates = [d for d in dates if d.weekday() < 5]
if len(dates) < N_DATES:
    # extend until we reach N_DATES weekdays
    d = dates[-1]
    while len(dates) < N_DATES:
        d += timedelta(days=1)
        if d.weekday() < 5: dates.append(d)

# ------------------ Construct latent true signal ------------------
# Factor structure per date, plus per-entity skill (alpha). We’ll create a
# per-date cross-sectional “score” s(t, i), and define the forward *excess* return
# as proportional to s(t, i). This gives us ground-truth RankIC ≈ 1 if the model learns s.

# per-date common factor (random walk)
f = np.cumsum(rng.normal(0, 0.1, size=len(dates)))

# per-entity alphas (stable across time)
alpha = rng.normal(0, 1.0, size=N_ENTS)

# sector tilts to test sector-neutralization later (optional)
sec_tilt_map = {f"S{k}": (k-3)*0.15 for k in range(1,6)}  # [-0.30..+0.30] in steps
sec_of = {entities[i]: sectors[i] for i in range(N_ENTS)}
tilt = np.array([sec_tilt_map[sec_of[e]] for e in entities])

# Score matrix s[t, i]
# Make it: s = 0.7*alpha + 0.3*f[t] + small noise, then sector tilt
S = []
for t, ft in enumerate(f):
    noise = rng.normal(0, 0.05, size=N_ENTS)
    s_t = 0.7*alpha + 0.3*ft + tilt + noise
    # Standardize cross-section so it's “rank-y”
    s_t = (s_t - s_t.mean()) / (s_t.std(ddof=1) + 1e-9)
    S.append(s_t)
S = np.array(S)  # shape [T, N_ENTS]

# Benchmark forward returns (market) ~ random walk increments; we’ll subtract for excess
mkt_fw = rng.normal(0, 0.02, size=len(dates))  # 20d-ish scale, arbitrary

# Forward returns for entities:
# Define *excess* forward return as proportional to score observed today: y_ex(t,i) = k*S[t,i] + eps
k = 0.02  # scale
eps = rng.normal(0, 0.005, size=S.shape)  # small noise
y_ex = k*S + eps  # this should yield RankIC ~ 1 vs the “true” S if learned

# If we also want raw forward returns, add back the market
y_raw = y_ex + mkt_fw[:, None]

# Shift forward by H to place at time t the forward H-day target (i.e., ret from t->t+H)
def fwd_shift(M, H):
    # M is [T, N]
    out = np.full_like(M, np.nan)
    T = M.shape[0]
    if H >= T: return out
    out[:-H,:] = M[H:,:]
    return out

y_ex_fwd  = fwd_shift(y_ex, H)
y_raw_fwd = fwd_shift(y_raw, H)
mkt_fwd   = fwd_shift(mkt_fw.reshape(-1,1), H).ravel()

# “Price-like” columns for some simple features (we don’t need to be perfect here)
# Build synthetic close by integrating daily returns constructed from S (for some weak correlation)
daily_r = 0.001*S + rng.normal(0,0.01, size=S.shape)
px0 = 100 + 10*rng.normal(size=N_ENTS)
px  = np.vstack([px0] + [px0]*(len(dates)-1)).astype(float)
for t in range(1, len(dates)):
    px[t,:] = px[t-1,:] * (1.0 + daily_r[t-1,:])
# basic trailing returns (arithmetic) to feed the model
def trail_ret(P, lag):
    out = np.full_like(P, np.nan)
    out[lag:,:] = (P[lag:,:] / P[:-lag,:]) - 1.0
    return out
ret_1d  = trail_ret(px, 1)
ret_5d  = trail_ret(px, 5)
ret_20d = trail_ret(px, 20)
ret_60d = trail_ret(px, 60)

# Some “volatility” proxy
def trailing_vol(P, lag):
    R = (P[1:,:]/P[:-1,:] - 1.0)
    out = np.full((P.shape[0], P.shape[1]), np.nan)
    for t in range(lag, P.shape[0]):
        out[t,:] = R[t-lag:t,:].std(axis=0, ddof=1)
    return out
vol_cc_20 = trailing_vol(px, 20)
vol_cc_60 = trailing_vol(px, 60)

# Minimal feature set that overlaps your registry; missing get dropped by run_model
# We'll include: ret_1d, ret_5d, ret_20d, ret_60d, vol_cc_20, vol_cc_60
# and a couple of dummy MA/oscillator cols so the script finds some.
def moving_avg(P, w):
    out = np.full_like(P, np.nan)
    for t in range(w-1, P.shape[0]):
        out[t,:] = P[t-w+1:t+1,:].mean(axis=0)
    return out
ma20 = moving_avg(px, 20)
px_over_ma20 = px / (ma20 + 1e-9)

# ------------------ Build Polars DataFrame ------------------
rows = []
for ti, d in enumerate(dates):
    for ei, e in enumerate(entities):
        rows.append((
            d, e, sec_of[e],
            px[ti,ei],
            ret_1d[ti,ei], ret_5d[ti,ei], ret_20d[ti,ei], ret_60d[ti,ei],
            vol_cc_20[ti,ei], vol_cc_60[ti,ei],
            px_over_ma20[ti,ei],
            y_raw_fwd[ti,ei], y_ex_fwd[ti,ei]
        ))
# add benchmark as a separate “entity” row per date for the market fwd target calc
for ti, d in enumerate(dates):
    rows.append((d, benchmark, "ETF", 100.0, np.nan, np.nan, np.nan, np.nan,
                 np.nan, np.nan, np.nan, mkt_fwd[ti], 0.0))

df = pl.DataFrame(
    rows,
    schema=[
        "asof","entity","sector",
        "close",
        "ret_1d","ret_5d","ret_20d","ret_60d",
        "vol_cc_20","vol_cc_60",
        "px_over_ma20",
        f"fwd{H}_ret",            # raw forward
        f"fwd{H}_ret_ex_mkt",     # excess forward vs market
    ],
    orient="row"
).with_columns(pl.col("asof").cast(pl.Date))

# ------------------ Write parquet ------------------
start = dates[0].isoformat()
end   = dates[-1].isoformat()
tgt_tag = "ex_mkt" if USE_EXCESS else "raw"
out_path = OUT_ROOT / f"daily_{start}_to_{end}_fwdH_{H}_tgt_{tgt_tag}.parquet"
df.write_parquet(out_path)
print(f"[ok] wrote toy features: {out_path}")
print(df.head(6))
print(df.tail(6))
