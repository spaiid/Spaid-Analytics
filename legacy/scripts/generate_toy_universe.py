# scripts/generate_toy_universe.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import polars as pl

# ---------------- Config ----------------
OUT_ROOT = Path(os.environ.get("SF_DATA_ROOT", "./_data")) / "features"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

H = int(os.environ.get("FWD_H", "20"))          # forward horizon
N_DATES = int(os.environ.get("N_DATES", "260")) # ~1y weekdays
SEED = int(os.environ.get("SEED", "42"))

rng = np.random.default_rng(SEED)

# Universe + sectors
UNIVERSE = ["AAPL","MSFT","TSLA","JPM","XOM","T"]
SECTOR_MAP = {"AAPL":"Tech","MSFT":"Tech","TSLA":"Auto","JPM":"Fin","XOM":"Energy","T":"Comm"}
BENCH = "SPY"

# Build weekdays
start = date(2020,1,2)
dates = []
d = start
while len(dates) < N_DATES:
    if d.weekday() < 5:
        dates.append(d)
    d += timedelta(days=1)

T = len(dates); N = len(UNIVERSE)

# ---------------- Latent structure ----------------
# Market factor (random walk) + sector tilts + entity alpha -> score S[t,i]
mkt_factor = np.cumsum(rng.normal(0, 0.08, size=T))
sec_tilts = {"Tech":0.25,"Auto":0.10,"Fin":-0.05,"Energy":-0.10,"Comm":0.05}
alpha = rng.normal(0, 1.0, size=N)

S = np.empty((T, N))
for t in range(T):
    noise = rng.normal(0, 0.07, size=N)
    tilt = np.array([sec_tilts[SECTOR_MAP[nm]] for nm in UNIVERSE])
    s_t = 0.7*alpha + 0.3*mkt_factor[t] + tilt + noise
    s_t = (s_t - s_t.mean()) / (s_t.std(ddof=1) + 1e-9)
    S[t] = s_t

# Excess forward returns (ground truth): y_ex[t,i] = k*S[t,i] + eps
k = 0.03
eps = rng.normal(0, 0.005, size=S.shape)
y_ex = k*S + eps

# Market (raw) forward return series (per-date)
mkt_fw = rng.normal(0, 0.015, size=T)

def fwd_shift(M, H):
    out = np.full_like(M, np.nan)
    if H < len(M):
        out[:-H] = M[H:]
    return out

# Forward targets at horizon H
y_ex_fwd  = fwd_shift(y_ex, H)                 # (T,N)
y_raw_fwd = y_ex_fwd + fwd_shift(mkt_fw, H)[:,None]  # raw = excess + market
mkt_fwd   = fwd_shift(mkt_fw, H)                       

# ------------- Price paths & simple features -------------
def synth_prices(T, N, base=100.0, shock=0.01):
    r = 0.001*S + rng.normal(0, shock, size=(T,N))
    P = np.empty((T,N)); P[0] = base * (1 + 0.02*rng.normal(size=N))
    for t in range(1,T):
        P[t] = P[t-1] * (1.0 + r[t-1])
    return P

PX = synth_prices(T, N, base=100.0, shock=0.012)
SPY = np.empty(T); SPY[0] = 320.0
spy_d = rng.normal(0, 0.008, size=T) + 0.002  # gentle drift for SPY
for t in range(1,T):
    SPY[t] = SPY[t-1] * (1.0 + spy_d[t-1])

def trail_ret(P, lag):
    out = np.full_like(P, np.nan, dtype=float)
    out[lag:] = (P[lag:] / P[:-lag]) - 1.0
    return out

def trailing_vol_from_price(P, lag):
    # close-to-close arithmetic vol
    R = np.empty_like(P); R[:] = np.nan
    R[1:] = (P[1:]/P[:-1]) - 1.0
    out = np.full_like(P, np.nan)
    for t in range(lag, P.shape[0]):
        out[t] = np.nanstd(R[t-lag:t], axis=0, ddof=1)
    return out

def moving_avg(P, w):
    out = np.full_like(P, np.nan)
    for t in range(w-1, P.shape[0]):
        out[t] = np.nanmean(P[t-w+1:t+1], axis=0)
    return out

ret_1d  = trail_ret(PX, 1)
ret_5d  = trail_ret(PX, 5)
ret_20d = trail_ret(PX, 20)
ret_60d = trail_ret(PX, 60)
vol_cc_20 = trailing_vol_from_price(PX, 20)
vol_cc_60 = trailing_vol_from_price(PX, 60)
ma20 = moving_avg(PX, 20)
px_over_ma20 = PX / (ma20 + 1e-9)

# ------------- Assemble rows -------------
rows = []
for ti, d in enumerate(dates):
    for j, nm in enumerate(UNIVERSE):
        rows.append((
            d, nm, SECTOR_MAP[nm],
            float(PX[ti,j]),
            ret_1d[ti,j], ret_5d[ti,j], ret_20d[ti,j], ret_60d[ti,j],
            vol_cc_20[ti,j], vol_cc_60[ti,j],
            px_over_ma20[ti,j],
            y_raw_fwd[ti,j], y_ex_fwd[ti,j]
        ))
# SPY benchmark row per date
for ti, d in enumerate(dates):
    rows.append((d, BENCH, "ETF",
        float(SPY[ti]),
        np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        mkt_fwd[ti], 0.0  # raw fwd return, excess = 0
    ))

df = pl.DataFrame(
    rows,
    schema=[
        "asof","entity","sector",
        "close",
        "ret_1d","ret_5d","ret_20d","ret_60d",
        "vol_cc_20","vol_cc_60",
        "px_over_ma20",
        f"fwd{H}_ret",
        f"fwd{H}_ret_ex_mkt",
    ],
    orient="row"
).with_columns(pl.col("asof").cast(pl.Date))

# Optional: light NA handling (the model runner also sanitizes)
def _nan_to_null(cols): 
    return [pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in cols]
feat_cols = ["ret_1d","ret_5d","ret_20d","ret_60d","vol_cc_20","vol_cc_60","px_over_ma20"]
df = df.with_columns(_nan_to_null(feat_cols + [f"fwd{H}_ret", f"fwd{H}_ret_ex_mkt"]))

start = dates[0].isoformat()
end   = dates[-1].isoformat()
out_path = OUT_ROOT / f"daily_{start}_to_{end}_fwdH_{H}_tgt_ex_mkt.parquet"
df.write_parquet(out_path)
print(f"[ok] wrote toy universe: {out_path}")
print(df.filter(pl.col("entity") != "SPY").head(8))
print(df.tail(8))
