"""Fase 1 - Audit dataset IoTMLCQ 2024 (Montería tilapia).
Usage: python3 -I scripts/audit.py data/raw.xlsx out/
"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, recall_score, f1_score

src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src)
R = {}

# ---------- 1. Struktur ----------
R["rows"], R["cols"] = df.shape
R["time_start"], R["time_end"] = str(df.Datetime.min()), str(df.Datetime.max())
isint = lambda s: (s - s.round()).abs() < 1e-9
genuine = isint(df["day"]) & isint(df["hour"]) & (df["hour"] <= 5)
# satu baris per (bulan-asli, day, hour) -> buang artefak duplikat
g = df[genuine].copy()
g["orig_month"] = np.floor(g["Month_Num"] + 1e-9).astype(int)
g = g.drop_duplicates(["orig_month", "day", "hour"])
R["genuine_obs"] = int(len(g))
R["genuine_share"] = round(len(g) / len(df), 3)
for c in ["Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)", "Survival Rate (%)"]:
    R[f"plateau_share::{c}"] = round(float((df[c].diff() == 0).mean()), 3)
for c in ["oxigeno_scaled", "ph", "turbidez"]:
    R[f"linear_interp_share::{c}"] = round(float((df[c].diff().diff().abs() < 1e-9).mean()), 3)

# ---------- 2. Label ----------
R["low_oxygen_alert_counts"] = df["Low Oxygen Alert"].value_counts().to_dict()
R["health_equals_thermal"] = bool(((df["Health Status"] == "At Risk") == (df["Thermal Risk Index"] == "High")).all())
hi = df["Thermal Risk Index"] == "High"
R["thermal_high_rule"] = f"Temperature > 28.0 (min High={df.loc[hi,'Temperature (°C)'].min():.3f}, max Normal={df.loc[~hi,'Temperature (°C)'].max():.3f})"
R["at_risk_by_month"] = df[df["Health Status"] == "At Risk"]["Month"].value_counts().to_dict()
R["oxy_auto_alternating"] = bool(((df["Oxygenation Automatic"] == "Yes") == (df.index % 2 == 0)).all())
monthly = (df.groupby("Month", sort=False)[["Survival Rate (%)", "Average Fish Weight (g)", "Disease Occurrence (Cases)",
          "Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)", "Precipitation (inches)"]]
          .agg(lambda s: s.mode()[0]).iloc[:6])
monthly.to_csv(f"{out}/monthly_values.csv")
R["survival_unique_months"] = int(monthly["Survival Rate (%)"].nunique())

# ---------- 3. Eksperimen inflasi metrik ----------
mnum = np.floor(df["Month_Num"] + 1e-9).clip(1, 6).astype(int)
# E1: regresi DO (kolom utama) dari parameter lain - split acak vs leave-one-month-out
X1 = df[["Temperature (°C)", "pH", "Turbidity (NTU)", "Precipitation (inches)", "Average Fish Weight (g)"]]
y1 = df["Dissolved Oxygen (mg/L)"]
Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(Xtr, ytr)
R["E1_DO_R2_random_split"] = round(r2_score(yte, rf.predict(Xte)), 4)
pred = np.zeros(len(df))
for m in range(1, 7):
    te = mnum == m
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X1[~te], y1[~te])
    pred[te] = rf.predict(X1[te])
R["E1_DO_R2_leave_one_month_out"] = round(r2_score(y1, pred), 4)

# E2: klasifikasi Health Status hanya dari kalender (Month_Num)
y2 = (df["Health Status"] == "At Risk").astype(int)
X2 = df[["Month_Num"]]
Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
p = clf.predict(Xte)
R["E2_health_from_month_only_random"] = {"acc": round(accuracy_score(yte, p), 4), "f1_atrisk": round(f1_score(yte, p), 4)}
te = mnum == 3
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X2[~te], y2[~te])
R["E2_health_recall_when_March_held_out"] = round(recall_score(y2[te], clf.predict(X2[te]), zero_division=0), 4)

# E3: forecasting oxigeno_scaled t+1 dari lag 1..6
def lagged(s, L=6):
    d = pd.DataFrame({f"lag{k}": s.shift(k) for k in range(1, L + 1)})
    d["y"] = s.shift(-1)
    return d.dropna()
d = lagged(df["oxigeno_scaled"].reset_index(drop=True))
Xtr, Xte, ytr, yte = train_test_split(d.drop(columns="y"), d.y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(Xtr, ytr)
R["E3_DOscaled_t+1_R2_all_rows_random"] = round(r2_score(yte, rf.predict(Xte)), 4)
gs = g.sort_values(["orig_month", "day", "hour"])["oxigeno_scaled"].reset_index(drop=True)
d = lagged(gs)
cut = int(len(d) * 0.8)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(d.iloc[:cut, :-1], d.y[:cut])
R["E3_DOscaled_t+1_R2_genuine_temporal"] = round(r2_score(d.y[cut:], rf.predict(d.iloc[cut:, :-1])), 4)
R["E3_persistence_R2_genuine_temporal"] = round(r2_score(d.y[cut:], d.lag1[cut:]), 4)

json.dump(R, open(f"{out}/audit_results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))

# ---------- Figur ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 1, figsize=(11, 9))
a = ax[0]
a.plot(df.Datetime, df["Temperature (°C)"], color="#2b6cb0", lw=1.4, label="Temperature (°C) – kolom utama")
a.fill_between(df.Datetime, 26.3, 28.5, where=hi, color="#e53e3e", alpha=0.18, label="Health Status = At Risk")
a.axhline(28, ls="--", color="#e53e3e", lw=0.8)
a.set_ylim(26.3, 28.5); a.set_ylabel("°C"); a.legend(loc="lower right", frameon=False)
a.set_title("(a) 'Sensor' suhu per jam = nilai bulanan konstan + ramp linear; At Risk = suhu > 28 °C (≈ bulan Maret)", loc="left")
b = ax[1]
w = df[(df.Datetime >= "2024-01-01") & (df.Datetime < "2024-01-08")]
gw = w[isint(w.day) & isint(w.hour) & (w.hour <= 5)]
b.plot(w.Datetime, w.oxigeno_scaled, color="#a0aec0", lw=1.2, label="nilai per jam di file")
b.scatter(gw.Datetime, gw.oxigeno_scaled, color="#2b6cb0", s=14, zorder=3, label="observasi asli (jam 00–05)")
b.set_ylabel("oxigeno_scaled"); b.legend(frameon=False, loc="lower left")
b.set_title("(b) Satu minggu: hanya 6 titik/hari yang asli, 18 jam sisanya garis lurus interpolasi", loc="left")
c = ax[2]
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
c.bar(labels, monthly["Survival Rate (%)"], color="#2b6cb0")
for i, v in enumerate(monthly["Survival Rate (%)"]):
    c.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
c.set_ylim(92, 97); c.set_ylabel("Survival Rate (%)")
c.set_title("(c) Variabel 'kelangsungan hidup' hanya punya 6 nilai (satu per bulan)", loc="left")
fig.tight_layout(); fig.savefig(f"{out}/fig_audit.png", dpi=160)
