"""Assemble the project notebook. Usage: python3 -I scripts/build_notebook.py scripts/simulator.py out/notebook.ipynb"""
import sys, re
import nbformat as nbf

sim_src = open(sys.argv[1]).read()
sim_core = sim_src.split('if __name__ == "__main__":')[0]
sim_core = sim_core.split('"""', 2)[2].lstrip()            # drop module docstring (explained in markdown)
sim_core = sim_core.replace("import sys, json\n", "")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Tugas Besar IoT — AIoT Early Warning Kelangsungan Hidup Tilapia
**Bagian 1: Audit dataset Montería + Digital Twin kolam tilapia**

Isi notebook:
1. **Setup & unduh data** langsung dari Kaggle (`kagglehub`) — tidak perlu upload ke Drive.
2. **Fase 1 – Audit dataset**: mengecek apakah data benar-benar pembacaan sensor per jam dan apakah labelnya layak dipakai.
3. **Eksperimen inflasi metrik**: menunjukkan kenapa akurasi tinggi di studi lain tidak bermakna.
4. **Digital twin kolam tilapia**: simulator per jam (lingkungan → manajemen → biologi → sensor IoT) yang dikalibrasi ke rata-rata bulanan Montería.
5. **Validasi simulator** dan penyimpanan dataset sintetis.

> Semua hasil acak memakai *seed* tetap, jadi angka yang keluar akan sama di laptop mana pun.
""")

code(r"""
# Jalankan sekali saja (Colab/laptop baru). Di Colab sebagian paket sudah terpasang.
%pip install -q kagglehub openpyxl scikit-learn matplotlib scipy
""")

code(r"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
ROOT = next((p for p in [".", ".."] if os.path.isdir(os.path.join(p, "src"))), ".")   # root repo bila dijalankan dari notebooks/
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
""")

md("## 1. Unduh data dari Kaggle")
code(r"""
import kagglehub

# Download latest version
# (TILAPIA_DATA_DIR hanya dipakai kalau ingin memakai folder lokal; biarkan kosong untuk unduh dari Kaggle)
path = os.environ.get("TILAPIA_DATA_DIR") or kagglehub.dataset_download("jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia")

print("Path to dataset files:", path)
""")
code(r"""
# Cari file data di folder unduhan (bisa .xlsx atau .csv)
files = [f for ext in ("xlsx", "xls", "csv") for f in glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)]
print("File ditemukan:", files)
assert files, "Tidak ada file .xlsx/.csv di folder dataset"
DATA_FILE = files[0]
df = pd.read_csv(DATA_FILE) if DATA_FILE.endswith(".csv") else pd.read_excel(DATA_FILE)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)
print(df.shape)
df.head()
""")

md(r"""
## 2. Fase 1 — Audit dataset
### 2.1 Struktur umum
Dataset diklaim berisi pembacaan sensor **per jam** Januari–Juni 2024. Kita cek jumlah nilai unik, kontinuitas waktu, dan missing value.
""")
code(r"""
print("Rentang waktu :", df.Datetime.min(), "→", df.Datetime.max())
print("Selisih waktu :", df.Datetime.diff().value_counts().to_dict())
print("Missing value :", int(df.isna().sum().sum()))
df.nunique().to_frame("n_unique").T
""")

md(r"""
### 2.2 Apakah kolom sensor benar-benar berubah tiap jam?
Kalau suhu/DO/pH/turbiditas adalah pembacaan sensor, nilainya seharusnya berubah hampir setiap jam.
Kita hitung porsi baris yang **sama persis** dengan jam sebelumnya (plateau) dan porsi baris yang berada di **garis lurus** (selisih kedua = 0 → hasil interpolasi linear).
""")
code(r"""
main_cols = ["Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)", "Survival Rate (%)"]
hidden_cols = ["oxigeno_scaled", "ph", "turbidez"]
audit = {}
for c in main_cols:
    audit[f"plateau_share | {c}"] = round(float((df[c].diff() == 0).mean()), 3)
for c in hidden_cols:
    audit[f"linear_interp_share | {c}"] = round(float((df[c].diff().diff().abs() < 1e-9).mean()), 3)
pd.Series(audit)
""")

md(r"""
### 2.3 Berapa observasi yang asli?
Kolom `day` dan `hour` ternyata ikut terinterpolasi (mis. `hour = 4.7368`). Observasi asli adalah baris dengan `day` dan `hour` bilangan bulat dan `hour` 0–5.
""")
code(r"""
isint = lambda s: (s - s.round()).abs() < 1e-9
genuine_mask = isint(df["day"]) & isint(df["hour"]) & (df["hour"] <= 5)
g = df[genuine_mask].copy()
g["orig_month"] = np.floor(g["Month_Num"] + 1e-9).astype(int)
g = g.drop_duplicates(["orig_month", "day", "hour"]).sort_values(["orig_month", "day", "hour"])
audit["genuine_obs"] = len(g)
audit["genuine_share"] = round(len(g) / len(df), 3)
print(f"Observasi asli: {len(g)} dari {len(df)} baris ({len(g)/len(df):.0%})")

# Dinamika observasi asli: autokorelasi & rata-rata bulanan
for c in hidden_cols:
    s = g[c].reset_index(drop=True)
    print(f"{c:15s} acf lag1={s.autocorr(1):+.3f}  lag6={s.autocorr(6):+.3f}  rata-rata bulanan={g.groupby('orig_month')[c].mean().round(3).tolist()}")
print("\nKorelasi oxigeno_scaled vs kolom DO utama:", round(df[["oxigeno_scaled", "Dissolved Oxygen (mg/L)"]].corr().iloc[0, 1], 3))
""")

md("### 2.4 Audit label (`Low Oxygen Alert`, `Thermal Risk Index`, `Health Status`, `Oxygenation Automatic`)")
code(r"""
hi = df["Thermal Risk Index"] == "High"
audit["low_oxygen_alert_counts"] = df["Low Oxygen Alert"].value_counts().to_dict()
audit["health_status_identik_thermal_risk"] = bool(((df["Health Status"] == "At Risk") == hi).all())
audit["aturan_thermal_high"] = f"Temperature > 28.0 (min High={df.loc[hi,'Temperature (°C)'].min():.3f}, max Normal={df.loc[~hi,'Temperature (°C)'].max():.3f})"
audit["at_risk_per_bulan"] = df.loc[df["Health Status"] == "At Risk", "Month"].value_counts().to_dict()
audit["oxygenation_automatic_selang_seling"] = bool(((df["Oxygenation Automatic"] == "Yes") == (df.index % 2 == 0)).all())
for k in ["low_oxygen_alert_counts", "health_status_identik_thermal_risk", "aturan_thermal_high", "at_risk_per_bulan", "oxygenation_automatic_selang_seling"]:
    print(f"{k}: {audit[k]}")
""")

md("### 2.5 Nilai bulanan (satu-satunya informasi survival yang tersedia)")
code(r"""
monthly = (df.groupby("Month", sort=False)[["Survival Rate (%)", "Average Fish Weight (g)", "Disease Occurrence (Cases)",
            "Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)", "Precipitation (inches)"]]
           .agg(lambda s: s.mode()[0]).iloc[:6])
display(monthly)
for c in ["Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)", "Precipitation (inches)"]:
    r, p = spearmanr(monthly[c], monthly["Survival Rate (%)"])
    print(f"Spearman {c:25s} vs survival: rho={r:+.2f}, p={p:.3f}  (n=6)")
""")

md(r"""
## 3. Eksperimen inflasi metrik (kenapa R² ≈ 0,999 tidak bermakna)
- **E1** – regresi DO (kolom utama) dengan RF + split acak vs *leave-one-month-out*.
- **E2** – klasifikasi `Health Status` hanya dari nomor bulan.
- **E3** – forecasting `oxigeno_scaled` t+1: semua baris + split acak vs observasi asli + split waktu.
""")
code(r"""
RF = dict(n_estimators=100, max_depth=10, random_state=42)
mnum = np.floor(df["Month_Num"] + 1e-9).clip(1, 6).astype(int)

# E1
X1 = df[["Temperature (°C)", "pH", "Turbidity (NTU)", "Precipitation (inches)", "Average Fish Weight (g)"]]; y1 = df["Dissolved Oxygen (mg/L)"]
Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.2, random_state=42)
audit["E1_R2_DO_split_acak"] = round(r2_score(yte, RandomForestRegressor(**RF).fit(Xtr, ytr).predict(Xte)), 4)
pred = np.zeros(len(df))
for m in range(1, 7):
    te = mnum == m
    pred[te] = RandomForestRegressor(**RF).fit(X1[~te], y1[~te]).predict(X1[te])
audit["E1_R2_DO_leave_one_month_out"] = round(r2_score(y1, pred), 4)

# E2
y2 = (df["Health Status"] == "At Risk").astype(int)
Xtr, Xte, ytr, yte = train_test_split(df[["Month_Num"]], y2, test_size=0.2, random_state=42, stratify=y2)
p = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xte)
audit["E2_health_dari_bulan_saja"] = {"acc": round(accuracy_score(yte, p), 4), "f1_at_risk": round(f1_score(yte, p), 4)}

# E3
def lagged(s, L=6):
    d = pd.DataFrame({f"lag{k}": s.shift(k) for k in range(1, L + 1)}); d["y"] = s.shift(-1); return d.dropna()
d = lagged(df["oxigeno_scaled"].reset_index(drop=True))
Xtr, Xte, ytr, yte = train_test_split(d.drop(columns="y"), d.y, test_size=0.2, random_state=42)
audit["E3_R2_semua_baris_split_acak"] = round(r2_score(yte, RandomForestRegressor(**RF).fit(Xtr, ytr).predict(Xte)), 4)
d = lagged(g["oxigeno_scaled"].reset_index(drop=True)); cut = int(len(d) * 0.8)
rf = RandomForestRegressor(**RF).fit(d.iloc[:cut, :-1], d.y[:cut])
audit["E3_R2_observasi_asli_split_waktu"] = round(r2_score(d.y[cut:], rf.predict(d.iloc[cut:, :-1])), 4)
audit["E3_R2_persistence_observasi_asli"] = round(r2_score(d.y[cut:], d.lag1[cut:]), 4)
{k: v for k, v in audit.items() if k.startswith("E")}
""")

md("### 3.1 Figur audit")
code(r"""
fig, ax = plt.subplots(3, 1, figsize=(11, 9))
a = ax[0]
a.plot(df.Datetime, df["Temperature (°C)"], color="#2b6cb0", lw=1.4, label="Temperature (°C) – kolom utama")
a.fill_between(df.Datetime, 26.3, 28.5, where=hi, color="#e53e3e", alpha=0.18, label="Health Status = At Risk")
a.axhline(28, ls="--", color="#e53e3e", lw=0.8); a.set_ylim(26.3, 28.5); a.set_ylabel("°C"); a.legend(loc="lower right", frameon=False)
a.set_title("(a) 'Sensor' suhu per jam = nilai bulanan konstan + ramp linear; At Risk = suhu > 28 °C (≈ bulan Maret)", loc="left")
b = ax[1]
w = df[(df.Datetime >= "2024-01-01") & (df.Datetime < "2024-01-08")]
gw = w[isint(w.day) & isint(w.hour) & (w.hour <= 5)]
b.plot(w.Datetime, w.oxigeno_scaled, color="#a0aec0", lw=1.2, label="nilai per jam di file")
b.scatter(gw.Datetime, gw.oxigeno_scaled, color="#2b6cb0", s=14, zorder=3, label="observasi asli (jam 00–05)")
b.set_ylabel("oxigeno_scaled"); b.legend(frameon=False, loc="lower left")
b.set_title("(b) Satu minggu: hanya 6 titik/hari yang asli, 18 jam sisanya garis lurus interpolasi", loc="left")
c = ax[2]; labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
c.bar(labels, monthly["Survival Rate (%)"], color="#2b6cb0")
for i, v in enumerate(monthly["Survival Rate (%)"]): c.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
c.set_ylim(92, 97); c.set_ylabel("Survival Rate (%)")
c.set_title("(c) Variabel 'kelangsungan hidup' hanya punya 6 nilai (satu per bulan)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_audit.png", dpi=160); plt.show()
json.dump(audit, open(f"{OUT}/audit_results.json", "w"), indent=2, default=str)
""")

md(r"""
**Kesimpulan audit:** hanya ±23% observasi yang asli; kolom sensor utama adalah nilai bulanan; `Health Status` = suhu bulanan > 28 °C (praktis = bulan Maret, justru bulan dengan survival tertinggi); `Low Oxygen Alert` selalu *Safe*; survival hanya 6 nilai.
Dataset ini **tidak cukup** untuk memprediksi kelangsungan hidup tilapia secara kredibel, sehingga dipakai sebagai **jangkar kalibrasi** untuk digital twin di bawah.
""")

md(r"""
## 4. Digital twin kolam tilapia (v0)
Simulator per jam untuk banyak kolam (Monte-Carlo), dengan 4 lapisan:

| Lapisan | Isi |
|---|---|
| Lingkungan | suhu air (siklus harian, musim hujan/kemarau, gelombang panas), fitoplankton (bloom & die-off), DO (neraca massa: fotosintesis − respirasi + reaerasi + aerator), pH (mengikuti fotosintesis), turbiditas (limpasan hujan) |
| Manajemen | kepadatan tebar, aerator otomatis histeresis (ON < 4 mg/L, OFF > 6 mg/L) dan kerusakan aerator |
| Biologi | hazard kematian per jam dari DO, suhu, pH, dan stres kronis (penyakit, tertunda 48 jam) |
| Sensor IoT | noise, drift pH (kalibrasi ulang tiap 15 hari), fouling DO, spike, data hilang, gateway mati |

**Jangkar literatur (dicatat di `PARAMS`)**: TLM-24 jam DO 1,41 mg/L (lab) dan toleransi 0,1–0,5 mg/L dengan akses permukaan (FAO) → nilai efektif 0,8 mg/L (diuji sensitivitasnya nanti); rentang suhu optimal 19,5–32,1 °C; CTmax 41,25–44,5 °C; pH tidak letal ≥ 3,5, layak 5,5–9,0.
**Kalibrasi**: rata-rata bulanan suhu, pH, turbiditas Montería; latar kematian bulanan 3,5%.
""")
code(sim_core)

md("### 4.1 Jalankan simulasi (200 kolam × 4.368 jam ≈ 15–30 detik)")
code(r"""
N_PONDS, SEED = 200, 42
rng = np.random.default_rng(SEED)
sim = pd.concat([simulate_pond(i, rng, PARAMS) for i in range(N_PONDS)], ignore_index=True)
print(sim.shape)
sim.head()
""")

md(r"""
## 5. Validasi simulator
Cek: (a) pola DO harian sesuai literatur (minimum menjelang fajar, maksimum sore), (b) rata-rata bulanan suhu & pH vs Montería, (c) contoh kejadian kematian massal, (d) survival bulanan vs Montería.
""")
code(r"""
d = sim.copy(); d["m"] = d.datetime.dt.month; d["h"] = d.datetime.dt.hour
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
ref = {"T": PARAMS["T_month"], "pH": PARAMS["pH_month"], "surv": monthly["Survival Rate (%)"].round(2).tolist()}
gm = d.groupby(["pond_id", "m"]).agg(start=("alive", "first"), dead=("deaths", "sum")); gm["surv"] = 100 * (1 - gm.dead / gm.start)
val = {
    "sim_T_month": d.groupby("m").true_temp_c.mean().round(2).tolist(), "ref_T_month": ref["T"],
    "sim_pH_month": d.groupby("m").true_ph.mean().round(2).tolist(), "ref_pH_month": ref["pH"],
    "sim_surv_overall_mean": round(gm.surv.mean(), 2), "ref_surv_overall_mean": round(float(np.mean(ref["surv"])), 2),
    "pond_months_below_90pct": round((gm.surv < 90).mean(), 3),
    "DO_hour_min": int(d.groupby("h").true_do_mgl.mean().idxmin()), "DO_hour_max": int(d.groupby("h").true_do_mgl.mean().idxmax()),
    "DO_dawn_mean": round(d[d.h == 5].true_do_mgl.mean(), 2), "DO_15h_mean": round(d[d.h == 15].true_do_mgl.mean(), 2),
    "MAE_T_month": round(float(np.mean(np.abs(np.array(d.groupby('m').true_temp_c.mean()) - ref["T"]))), 3),
    "MAE_pH_month": round(float(np.mean(np.abs(np.array(d.groupby('m').true_ph.mean()) - ref["pH"]))), 3),
    "MAE_sensor_DO": round((d.sensor_do_mgl - d.true_do_mgl).abs().mean(), 3),
}
json.dump(val, open(f"{OUT}/sim_validation.json", "w"), indent=1)
pd.Series(val)
""")
code(r"""
fig = plt.figure(figsize=(12, 9.5)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.15, 1]); x = np.arange(6)
ax = fig.add_subplot(gs[0, 0])
q = d.groupby("h").true_do_mgl.quantile([0.1, 0.5, 0.9]).unstack()
ax.fill_between(q.index, q[0.1], q[0.9], color="#90cdf4", alpha=0.6, label="P10–P90 (semua kolam)")
ax.plot(q.index, q[0.5], color="#2b6cb0", lw=2, label="median")
ax.axhline(4, ls="--", color="#dd6b20", lw=0.9); ax.text(0.3, 4.15, "batas stres 4 mg/L", color="#dd6b20", fontsize=8)
ax.set_xlabel("jam"); ax.set_ylabel("DO (mg/L)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(a) Siklus harian DO: minimum menjelang fajar, maksimum sore", loc="left")

ax = fig.add_subplot(gs[0, 1])
ax.plot(x, ref["T"], "o-", color="#e53e3e", label="Montería (rata-rata bulanan)")
ax.plot(x, val["sim_T_month"], "s--", color="#2b6cb0", label="simulasi")
ax.set_xticks(x, MON); ax.set_ylabel("suhu air (°C)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(b) Kalibrasi suhu bulanan terhadap Montería", loc="left")

ax = fig.add_subplot(gs[1, :])
ev = d.groupby("pond_id").deaths.sum().sort_values(); pid = ev.index[int(len(ev) * 0.9)]
p = d[d.pond_id == pid]; worst = p.set_index("datetime").deaths.resample("D").sum().idxmax()
w = p[(p.datetime >= worst - pd.Timedelta(days=6)) & (p.datetime <= worst + pd.Timedelta(days=3))]
ax.plot(w.datetime, w.true_do_mgl, color="#2b6cb0", lw=1.5, label="DO aktual")
ax.plot(w.datetime, w.sensor_do_mgl, color="#a0aec0", lw=0.8, label="DO terbaca sensor (noise/drift/missing)")
ax.axhline(PARAMS["DO_TLM24"], ls="--", color="#e53e3e", lw=0.8); ax.set_ylabel("DO (mg/L)")
ax2 = ax.twinx(); ax2.bar(w.datetime, w.deaths, width=0.04, color="#e53e3e", alpha=0.7, label="ikan mati / jam")
ax2.set_ylabel("ikan mati / jam", color="#e53e3e"); ax2.spines["top"].set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"(c) Contoh kejadian: kolam {pid} — DO turun beberapa malam berturut-turut sebelum kematian massal", loc="left")

ax = fig.add_subplot(gs[2, :])
ax.boxplot([gm.xs(m, level="m").surv.clip(lower=60) for m in range(1, 7)], positions=x, widths=0.5,
           flierprops=dict(markersize=2, alpha=0.4))
ax.plot(x, ref["surv"], "o", color="#e53e3e", ms=7, label="Montería")
ax.plot(x, gm.surv.groupby("m").mean().values, "s", color="#2b6cb0", ms=6, label="rata-rata simulasi")
ax.set_xticks(x, MON); ax.set_ylabel("survival bulanan (%)"); ax.set_ylim(59, 100); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("(d) Distribusi survival bulanan per kolam (nilai <60% dipotong ke 60) vs Montería", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sim_validation.png", dpi=160); plt.show()
""")

md(r"""
## 6. Label awal & simpan dataset sintetis
Label early warning: **kematian > 0,5% stok dalam 24 jam ke depan** (≈ 4× kematian latar). Horizon 6/12 jam dibuat di fase berikutnya.
""")
code(r"""
fut24 = sim.groupby("pond_id").deaths.transform(lambda s: s[::-1].rolling(24, min_periods=24).sum()[::-1].shift(-1))
sim["y_excess_mort_24h"] = (fut24 / sim.alive.replace(0, np.nan) > 0.005).astype("Int8")
y = sim["y_excess_mort_24h"].fillna(0).astype(bool)
episodes = int((y & ~y.groupby(sim.pond_id).shift(1, fill_value=False)).sum())
print(f"Prevalensi label: {y.mean():.2%} | episode: {episodes} | kolam dengan kejadian: {sim.assign(y=y).groupby('pond_id').y.any().sum()}")
sim.to_csv(f"{OUT}/sim_ponds.csv.gz", index=False, compression="gzip")
json.dump(PARAMS, open(f"{OUT}/sim_params.json", "w"), indent=1)
print("Tersimpan di folder:", OUT, os.listdir(OUT))
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[2])
print("cells:", len(cells))
