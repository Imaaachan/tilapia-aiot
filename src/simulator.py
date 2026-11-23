"""Tilapia pond digital twin (v0) — hourly, Monte-Carlo over many ponds.

Layers
  1. Environment  : water temperature, phytoplankton, DO (mass balance), pH, turbidity
  2. Management   : stocking density, automatic aeration (on/off + failures)
  3. Biology      : hourly mortality hazard from DO / temperature / pH / chronic stress (disease)
  4. IoT sensors  : noise, drift (recalibration every 15 d), spikes, dropouts

Literature anchors (see PARAMS["sources"]) — every biological threshold is cited.
Calibration anchor: monthly means of the Montería IoTMLCQ 2024 dataset (temperature, pH, turbidity).

Usage: python3 -I scripts/simulator.py <n_ponds> <seed> <out_dir>
"""
import sys, json
import numpy as np, pandas as pd

PARAMS = {
    # --- calibration to Montería monthly means (Jan..Jun 2024) ---
    "T_month": [27.47, 26.50, 28.35, 27.70, 26.60, 27.45],
    "pH_month": [7.98, 7.34, 7.97, 7.91, 7.89, 7.88],
    "turb_month": [3.30, 2.79, 2.79, 3.98, 3.19, 3.71],
    "rain_prob_month": [0.10, 0.07, 0.10, 0.35, 0.55, 0.50],   # dry Dec–Mar, rainy Apr/May–Nov
    "heatwave_prob_month": [0.02, 0.03, 0.04, 0.02, 0.0, 0.0],  # hot season Feb–Apr
    # --- environment ---
    "T_diel_amp": (0.8, 1.6),            # °C, afternoon peak ~15:00
    "Pmax": (1.3, 1.9),                  # mg O2 L^-1 h^-1 at chl=1, full light
    "r_plankton": 0.16, "r_sed": 0.07,   # mg L^-1 h^-1 at 20 °C
    "r_fish_per_kgm3": 0.045,            # mg L^-1 h^-1 per kg fish m^-3 at 20 °C
    "theta": 1.07, "k_reaer": (0.03, 0.06),
    "pH_beta": 0.9, "pH_calib_offset": 0.17,                      # pH rise per unit relative O2 supersaturation
    # --- management ---
    "stock_density_kg_m3": (1.0, 3.5), "stock_density_noaer_kg_m3": (0.6, 1.6),
    "aerator_prob": 0.80, "aerator_on_below": 4.0, "aerator_off_above": 6.0,
    "aerator_kla": (0.25, 0.45), "aerator_fail_prob_day": 0.006, "aerator_fail_days": (1, 4),
    # --- biology (hazard per hour) ---
    "N0": 10000,
    "h0_month": 0.035,                   # background monthly mortality (Montería survival 92.9–96.3 %)
    "DO_TLM24_lab": 1.41,                # 24-h median tolerance limit, lab N2-bubbling (Magid & Babiker 1975)
    "DO_TLM24": 0.80,                    # effective pond value: surface access lets tilapia tolerate 0.1–0.5 mg/L (FAO ARAC/87/WP/10)
                                         # -> uncertain parameter, varied in sensitivity analysis (0.5–1.41)
    "DO_TLM_temp_slope": 0.15,           # mg/L per °C above 30 °C (heat lowers hypoxia tolerance)
    "DO_scale": 0.25,
    "T_lethal": 39.0, "T_scale": 0.8,    # acute upper lethal zone (CTmax 41–44.5 °C)
    "T_opt_hi": 32.1,                    # upper bound optimal range (19.5–32.1 °C)
    "pH_lethal_low": 3.5, "pH_ok": (5.5, 9.0),
    "stress_halflife_h": 72, "disease_lag_h": 48, "k_disease": 0.0015,
    # --- sensors ---
    "noise": {"T": 0.10, "DO": 0.15, "pH": 0.05, "turb_rel": 0.05},
    "pH_drift_per_h": (-0.0012, 0.0012), "DO_fouling_per_h": (-0.0015, 0.0),
    "recal_every_h": 15 * 24, "spike_prob": 0.002, "dropout_prob": 0.01,
    "outage_prob_day": 0.01, "outage_hours": (3, 24),
}


def do_sat(T):
    """Freshwater DO saturation (mg/L) at sea level."""
    return 14.652 - 0.41022 * T + 0.007991 * T**2 - 0.000077774 * T**3


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_pond(pid, rng, P, start="2024-01-01", end="2024-07-01"):
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(idx)
    hour = idx.hour.values
    day = (np.arange(n) // 24)
    nd = day.max() + 1
    mon = idx.month.values - 1

    u = lambda a: rng.uniform(*a)
    T_amp, Pmax, k_re = u(P["T_diel_amp"]), u(P["Pmax"]), u(P["k_reaer"])
    has_aer = rng.random() < P["aerator_prob"]
    # farms without aeration stock less (semi-intensive practice)
    dens = u(P["stock_density_kg_m3"]) if has_aer else u(P["stock_density_noaer_kg_m3"])
    kla = u(P["aerator_kla"]) if has_aer else 0.0

    # smooth monthly baselines (interpolate between month mid-points)
    # hours since start (unit-safe; pandas may store datetime64[us])
    t0 = idx[0]
    mid = np.array([(pd.Timestamp(f"2024-{m+1:02d}-15") - t0) / pd.Timedelta(hours=1) for m in range(6)])
    tv = np.arange(n, dtype=float)
    # calibration: compensate mean rain cooling / mean photosynthetic pH deficit so monthly means match Montería
    T_base = np.interp(tv, mid, np.array(P["T_month"]) + 0.8 * np.array(P["rain_prob_month"]))
    pH_base = np.interp(tv, mid, np.array(P["pH_month"]) + P["pH_calib_offset"] + 0.25 * np.array(P["rain_prob_month"]))
    turb_base = np.interp(tv, mid, P["turb_month"])
    mday = np.array([idx[d * 24].month - 1 for d in range(nd)])

    # daily weather: rain, cloud, heatwave (multi-day)
    rain = rng.random(nd) < np.array(P["rain_prob_month"])[mday]
    cloud = np.clip(np.where(rain, rng.uniform(0.25, 0.6, nd), rng.uniform(0.75, 1.0, nd)), 0, 1)
    heat = np.zeros(nd)
    d = 0
    while d < nd:
        if rng.random() < P["heatwave_prob_month"][mday[d]]:
            L = rng.integers(3, 8); amp = rng.uniform(1.5, 4.0)
            heat[d:d + L] = amp * np.sin(np.linspace(0.2, np.pi - 0.2, len(heat[d:d + L])))
            d += L
        d += 1
    T_anom = np.zeros(nd)
    for i in range(1, nd):
        T_anom[i] = 0.7 * T_anom[i - 1] + rng.normal(0, 0.35)
    T_day = T_anom + heat - 0.8 * rain
    T = T_base + T_day[day] + T_amp * np.sin(2 * np.pi * (hour - 9) / 24)

    # phytoplankton: mean-reverting with blooms and die-offs (die-off -> DO crash risk)
    chl = np.ones(nd)
    chl[0] = rng.uniform(0.8, 1.2)
    bloom_p = 0.008 + 0.008 * (dens - 1.0) / 2.5
    for i in range(1, nd):
        c = chl[i - 1] + 0.08 * (1.0 + 0.15 * (dens - 3) - chl[i - 1]) + rng.normal(0, 0.05)
        if rng.random() < bloom_p:
            c += rng.uniform(0.6, 1.4)                # bloom
        if chl[i - 1] > 1.8 and rng.random() < 0.035:
            c = rng.uniform(0.35, 0.6)                # bloom crash (die-off)
        chl[i] = np.clip(c, 0.15, 3.5)
    chl_h = chl[day]
    light = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * cloud[day]

    # aerator failures
    aer_fail = np.zeros(nd, bool)
    d = 0
    while d < nd:
        if has_aer and rng.random() < P["aerator_fail_prob_day"]:
            L = rng.integers(*P["aerator_fail_days"]); aer_fail[d:d + L] = True; d += L
        d += 1

    # --- DO mass balance (hourly Euler) + biology ---
    th = P["theta"]
    DO = np.zeros(n); aer = np.zeros(n, int)
    DO[0] = do_sat(T[0])
    N = np.zeros(n); deaths = np.zeros(n); haz = np.zeros(n); hcomp = np.zeros((n, 4))
    N_t = float(P["N0"])
    h0 = -np.log(1 - P["h0_month"]) / (30 * 24)
    a_stress = np.log(2) / P["stress_halflife_h"]
    S = 0.0; S_hist = np.zeros(n)
    pH_true = np.zeros(n)
    rain_h = rain[day]
    on = False
    for t in range(n):
        Tt = T[t]; sat = do_sat(Tt)
        biomass = dens * N_t / P["N0"]
        resp = (P["r_plankton"] * chl_h[t] + P["r_sed"] + P["r_fish_per_kgm3"] * biomass) * th ** (Tt - 20)
        prod = Pmax * chl_h[t] * light[t]
        prev = DO[t - 1] if t else DO[0]
        # hysteresis aerator control (the IoT actuator)
        if has_aer and not aer_fail[day[t]]:
            if prev < P["aerator_on_below"]: on = True
            elif prev > P["aerator_off_above"]: on = False
        else:
            on = False
        aer[t] = int(on)
        degas = 0.3 * max(0.0, prev - 1.6 * sat)          # bubble loss at strong supersaturation
        dDO = prod - resp + k_re * (sat - prev) + (kla * (sat - prev) if on else 0.0) - degas
        DOt = max(0.05, prev + dDO + rng.normal(0, 0.05))
        DO[t] = DOt
        # pH follows photosynthetic CO2 uptake (relative supersaturation)
        pH_true[t] = pH_base[t] + P["pH_beta"] * np.clip((DOt - sat) / sat, -0.8, 1.0) - (0.25 if rain_h[t] else 0.0)
        # ---- hazards ----
        tlm = P["DO_TLM24"] + P["DO_TLM_temp_slope"] * max(0.0, Tt - 30)
        lam24 = np.log(2) / 24
        h_do = 2 * lam24 * sig((tlm - DOt) / P["DO_scale"])
        h_T = 2 * lam24 * sig((Tt - P["T_lethal"]) / P["T_scale"])
        h_pH = 2 * lam24 * sig((P["pH_lethal_low"] - pH_true[t]) / 0.15)
        # chronic stress index (dimensionless), EWMA
        s_now = (max(0.0, (4.0 - DOt) / 4.0) + max(0.0, (Tt - P["T_opt_hi"]) / 4.0)
                 + max(0.0, P["pH_ok"][0] - pH_true[t]) / 1.5 + max(0.0, pH_true[t] - P["pH_ok"][1]) / 1.5)
        S = S * np.exp(-a_stress) + s_now * (1 - np.exp(-a_stress))
        S_hist[t] = S
        S_lag = S_hist[t - P["disease_lag_h"]] if t >= P["disease_lag_h"] else 0.0
        h_dis = P["k_disease"] * S_lag ** 1.5
        h = h0 + h_do + h_T + h_pH + h_dis
        haz[t] = h
        hcomp[t] = (h_do, h_T, h_pH, h_dis)
        dd = rng.binomial(int(N_t), 1 - np.exp(-h)) if N_t > 0 else 0
        N_t -= dd; deaths[t] = dd; N[t] = N_t

    turb = turb_base * (0.8 + 0.25 * chl_h)
    # rain run-off pulses decaying with tau ~ 24 h
    pulse = np.zeros(n)
    for t in range(1, n):
        add = rng.uniform(3, 20) if (rain_h[t] and hour[t] == 14) else 0.0
        pulse[t] = pulse[t - 1] * np.exp(-1 / 24) + add
    turb = turb + pulse

    # --- IoT sensor layer ---
    nz = P["noise"]
    recal = (np.arange(n) % P["recal_every_h"])
    pH_drift = u(P["pH_drift_per_h"]) * recal
    DO_foul = 1 + u(P["DO_fouling_per_h"]) * recal / 24
    s_T = T + rng.normal(0, nz["T"], n)
    s_DO = np.clip(DO * DO_foul + rng.normal(0, nz["DO"], n), 0, None)
    s_pH = pH_true + pH_drift + rng.normal(0, nz["pH"], n)
    s_turb = np.clip(turb * (1 + rng.normal(0, nz["turb_rel"], n)), 0, None)
    sens = {"T": s_T, "DO": s_DO, "pH": s_pH, "turb": s_turb}
    for k in sens:
        sp = rng.random(n) < P["spike_prob"]
        sens[k] = np.where(sp, sens[k] * rng.choice([0.0, 2.5], n), sens[k])
        drop = rng.random(n) < P["dropout_prob"]
        sens[k] = np.where(drop, np.nan, sens[k])
    outage = np.zeros(n, bool)                  # gateway outage -> all sensors missing
    for dd_ in np.where(rng.random(nd) < P["outage_prob_day"])[0]:
        s0 = dd_ * 24 + rng.integers(0, 24); outage[s0:s0 + rng.integers(*P["outage_hours"])] = True
    for k in sens:
        sens[k] = np.where(outage, np.nan, sens[k])

    return pd.DataFrame({
        "pond_id": pid, "datetime": idx,
        "sensor_temp_c": sens["T"], "sensor_do_mgl": sens["DO"], "sensor_ph": sens["pH"], "sensor_turb_ntu": sens["turb"],
        "aerator_on": aer, "rain_day": rain_h.astype(int),
        "true_temp_c": T, "true_do_mgl": DO, "true_ph": pH_true, "true_turb_ntu": turb, "chl_index": chl_h,
        "stress_index": S_hist, "hazard_per_h": haz,
        "h_do": hcomp[:, 0], "h_temp": hcomp[:, 1], "h_ph": hcomp[:, 2], "h_disease": hcomp[:, 3], "deaths": deaths, "alive": N,
        "stock_density_kg_m3": dens, "has_aerator": int(has_aer),
    })


if __name__ == "__main__":
    n_ponds, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    df = pd.concat([simulate_pond(i, rng, PARAMS) for i in range(n_ponds)], ignore_index=True)
    df.to_parquet(f"{out}/sim_ponds.parquet", index=False)
    json.dump({k: v for k, v in PARAMS.items()}, open(f"{out}/sim_params.json", "w"), indent=1)
    print(df.shape)
