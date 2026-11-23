"""Edge-computable features + multi-horizon labels.

Every feature uses only the last <=72 h of SENSOR readings and the aerator state the IoT node itself controls,
so it can be computed on an ESP32 with a 72-slot ring buffer per sensor.
"""
import numpy as np, pandas as pd

SENSORS = {"sensor_temp_c": "T", "sensor_do_mgl": "DO", "sensor_ph": "pH", "sensor_turb_ntu": "turb"}
VALID = {"T": (10, 45), "DO": (0, 25), "pH": (3, 11), "turb": (0, 300)}   # physically plausible ranges
HORIZONS = (6, 12, 24)
THR = 0.005                                                                  # >0.5 % of stock dies within horizon


def clean(d):
    """Edge preprocessing: drop implausible values (sensor error codes / spikes), hampel-like jump filter, ffill <=3 h."""
    out = d[["pond_id", "datetime", "aerator_on"]].copy()
    for col, k in SENSORS.items():
        s = d[col].where(d[col].between(*VALID[k]))
        med = s.groupby(d.pond_id).transform(lambda x: x.rolling(5, center=False, min_periods=2).median())
        jump = (s - med).abs() > {"T": 3, "DO": 4, "pH": 1.0, "turb": 50}[k]
        s = s.mask(jump)
        out[f"miss_{k}"] = s.isna().astype(np.int8)
        out[k] = s.groupby(d.pond_id).ffill(limit=3)
    return out


def make_features(c):
    g = c.groupby("pond_id")
    F = pd.DataFrame(index=c.index)
    hr = c.datetime.dt.hour
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    for k in ["T", "DO", "pH", "turb"]:
        F[k] = c[k]
    roll = lambda col, w, f: g[col].transform(lambda x: getattr(x.rolling(w, min_periods=max(2, w // 3)), f)())
    for L in (1, 3, 6, 12, 24):
        F[f"DO_lag{L}"] = g["DO"].shift(L)
    for L in (6, 24):
        F[f"T_lag{L}"] = g["T"].shift(L); F[f"pH_lag{L}"] = g["pH"].shift(L)
    F["DO_d1"] = F["DO"] - F["DO_lag1"]; F["DO_d6"] = F["DO"] - F["DO_lag6"]; F["DO_d24"] = F["DO"] - F["DO_lag24"]
    for w in (6, 24, 72):
        F[f"DO_mean{w}"] = roll("DO", w, "mean"); F[f"DO_min{w}"] = roll("DO", w, "min")
    F["DO_max24"] = roll("DO", 24, "max"); F["DO_max72"] = roll("DO", 72, "max"); F["DO_std24"] = roll("DO", 24, "std")
    F["DO_amp24"] = F["DO_max24"] - F["DO_min24"]
    F["DO_min24_lag24"] = g["DO"].shift(24).groupby(c.pond_id).transform(lambda x: x.rolling(24, min_periods=8).min())
    F["DO_max24_lag24"] = g["DO"].shift(24).groupby(c.pond_id).transform(lambda x: x.rolling(24, min_periods=8).max())
    F["DO_peak_trend"] = F["DO_max24"] - F["DO_max24_lag24"]          # falling afternoon peak = weakening photosynthesis
    F["DO_trough_trend"] = F["DO_min24"] - F["DO_min24_lag24"]        # falling dawn trough
    low = (c["DO"] < 4).astype(float); vlow = (c["DO"] < 2).astype(float)
    F["hrs_DO_lt4_24"] = low.groupby(c.pond_id).transform(lambda x: x.rolling(24, min_periods=1).sum())
    F["hrs_DO_lt4_72"] = low.groupby(c.pond_id).transform(lambda x: x.rolling(72, min_periods=1).sum())
    F["hrs_DO_lt2_72"] = vlow.groupby(c.pond_id).transform(lambda x: x.rolling(72, min_periods=1).sum())
    F["T_mean24"] = roll("T", 24, "mean"); F["T_max24"] = roll("T", 24, "max")
    F["pH_max24"] = roll("pH", 24, "max"); F["pH_min24"] = roll("pH", 24, "min")
    F["turb_mean24"] = roll("turb", 24, "mean"); F["turb_max24"] = roll("turb", 24, "max")
    F["aer_frac24"] = c.groupby("pond_id").aerator_on.transform(lambda x: x.rolling(24, min_periods=1).mean())
    F["aer_on"] = c["aerator_on"]
    F["miss6"] = c[[f"miss_{k}" for k in ["T", "DO", "pH", "turb"]]].sum(axis=1).groupby(c.pond_id).transform(
        lambda x: x.rolling(6, min_periods=1).sum())
    # temperature-adjusted oxygen: % saturation (sat curve is cheap to compute on MCU)
    sat = 14.652 - 0.41022 * c["T"] + 0.007991 * c["T"] ** 2 - 0.000077774 * c["T"] ** 3
    F["DO_pct_sat"] = 100 * c["DO"] / sat
    F["DO_min24_pct_sat"] = 100 * F["DO_min24"] / sat
    return F.astype(np.float32)


def make_labels(d, horizons=HORIZONS, thr=THR):
    Y = pd.DataFrame(index=d.index)
    alive = d.alive.replace(0, np.nan)
    for H in horizons:
        fut = d.groupby("pond_id").deaths.transform(lambda s: s[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1))
        Y[f"y{H}"] = (fut / alive > thr).astype(float).where(fut.notna())
    return Y
