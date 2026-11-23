"""Build notebooks/02_features_models.ipynb. Usage: python3 -I scripts/build_nb02.py notebooks/02_features_models.ipynb"""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Tugas Besar IoT — Bagian 2: Fitur Edge, Label Multi-Horizon, dan Model Early Warning

Lanjutan dari Bagian 1 (audit + digital twin). Isi notebook:
1. **Setup**: bekerja di folder repo lokal, atau otomatis `git clone` repo GitHub di Colab/laptop baru.
2. **Data**: memakai `outputs/sim_ponds.csv.gz` dari Bagian 1; kalau belum ada, simulasi dijalankan ulang (seed sama → data identik).
3. **Fase 3 – Fitur & label**: 46 fitur yang bisa dihitung di ESP32 (buffer 72 jam) + label kematian berlebih 6/12/24 jam.
4. **Fase 4 – Model**: baseline aturan, regresi logistik, Random Forest, LightGBM dengan **split per kolam + waktu** (bebas leakage).
5. **RQ2 – Inflasi metrik**: split acak vs split per kolam vs split per kolam + waktu.
6. **Figur & tabel** untuk paper.

> Waktu jalan penuh ±15–20 menit di CPU 2 core (Colab gratis). Set `QUICK = True` untuk uji cepat (±4 menit).
""")

code(r"""
import os, sys, subprocess
REPO_URL = "https://github.com/USERNAME/tilapia-aiot.git"   # ganti USERNAME setelah repo dibuat

# cari root repo (folder yang berisi src/)
ROOT = next((p for p in [".", "..", "tilapia-aiot"] if os.path.isdir(os.path.join(p, "src"))), None)
if ROOT is None:                                  # Colab / laptop baru → clone dari GitHub
    subprocess.run(["git", "clone", REPO_URL], check=True); ROOT = "tilapia-aiot"
ROOT = os.path.abspath(ROOT); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
print("Root repo:", ROOT)
""")
code(r"""
%pip install -q lightgbm scikit-learn pandas numpy matplotlib pyarrow
""")
code(r"""
import json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from simulator import simulate_pond, PARAMS
from features import clean, make_features, make_labels, HORIZONS, THR
from modeling import evaluate, make_model, downsample, best_f1_threshold, CUT_TEST, CUT_VAL

QUICK = False                       # True = 3 fold & 1 horizon tambahan (untuk uji cepat)
N_SPLITS = 3 if QUICK else 5
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
""")

md("## 1. Data simulasi (dari Bagian 1, atau dibuat ulang)")
code(r"""
SIM = f"{OUT}/sim_ponds.csv.gz"
if os.path.exists(SIM):
    d = pd.read_csv(SIM, parse_dates=["datetime"])
else:
    print("File Bagian 1 tidak ditemukan → menjalankan simulasi ulang (200 kolam, seed 42) ...")
    rng = np.random.default_rng(42)
    d = pd.concat([simulate_pond(i, rng, PARAMS) for i in range(200)], ignore_index=True)
d = d.sort_values(["pond_id", "datetime"]).reset_index(drop=True)
print(d.shape, "| kolam:", d.pond_id.nunique(), "|", d.datetime.min(), "→", d.datetime.max())
""")

md(r"""
## 2. Fase 3 — Preprocessing edge, fitur, dan label
**Preprocessing (dijalankan di node IoT):** buang nilai di luar rentang fisik (mis. kode error sensor), filter lonjakan terhadap median 5 jam, isi data hilang maksimal 3 jam (*forward-fill*).

**Fitur (hanya dari 72 jam terakhir bacaan sensor + status aerator):** nilai sekarang, lag 1–24 jam, selisih DO, rata-rata/min/maks/std bergulir DO (6/24/72 jam), tren puncak sore & lembah fajar DO, jumlah jam DO < 4 dan < 2 mg/L, % saturasi DO (dihitung dari suhu), statistik suhu/pH/turbiditas 24 jam, fraksi aerator menyala, jumlah data hilang.
Kolom `true_*`, `alive`, `deaths`, dan `hazard` **tidak pernah** dipakai sebagai fitur.

**Label:** `y{H} = 1` jika ikan yang mati dalam H jam ke depan > 0,5% stok (≈ 4× kematian latar harian).
""")
code(r"""
t0 = time.time()
c = clean(d); X = make_features(c); Y = make_labels(d)
meta = d[["pond_id", "datetime", "alive", "deaths", "true_do_mgl", "has_aerator"]].copy()
print(f"Fitur: {X.shape[1]} kolom | waktu {time.time()-t0:.1f} s")
print("Prevalensi label:", Y.mean().round(4).to_dict())
X.describe().T[["mean", "min", "max"]].round(2).head(12)
""")

md(r"""
## 3. Fase 4 — Protokol evaluasi bebas leakage
- **Split per kolam (GroupKFold, 5 fold):** kolam yang diuji tidak pernah dilihat saat training.
- **Split waktu:** training Jan–Apr, pengujian Mei–Jun.
- **Threshold alarm** dipilih memaksimalkan F1 di **April kolam training** (model dilatih Jan–Mar) → tidak ada informasi data uji yang bocor.
- **Metrik:** PR-AUC (utama, karena kelas positif langka), ROC-AUC, precision/recall/F1 pada threshold, **event recall** (kejadian yang didahului alarm ≤ 48 jam sebelum onset kematian), **median lead time**, dan **jam alarm palsu per kolam-bulan**.
""")
code(r"""
horizons = (24,) if QUICK else HORIZONS
res = []
for H in horizons:
    t0 = time.time()
    res.append(evaluate(X, Y, meta, H, n_splits=N_SPLITS, protocol="pond_time"))
    print(f"H={H} selesai ({time.time()-t0:.0f} s)")
res = pd.concat(res); res.to_csv(f"{OUT}/results_pond_time.csv", index=False)
cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "event_recall", "median_lead_h", "false_alarm_h_per_pond_month"]
summary = res.groupby(["horizon", "model"])[cols].agg(["mean", "std"]).round(3)
summary
""")

md("### 3.1 Tabel ringkas (mean ± sd antar fold) — siap untuk paper")
code(r"""
def pm(g, c): return f"{g[c].mean():.3f} ± {g[c].std():.3f}"
tab = (res.groupby(["horizon", "model"])
          .apply(lambda g: pd.Series({"PR-AUC": pm(g, "pr_auc"), "ROC-AUC": pm(g, "roc_auc"), "F1": pm(g, "f1"),
                                      "Event recall": pm(g, "event_recall"), "Lead time (h)": f"{g.median_lead_h.median():.1f}",
                                      "False alarm h / pond-month": f"{g.false_alarm_h_per_pond_month.mean():.2f}",
                                      "Prevalence": f"{g.prevalence.mean():.4f}"}), include_groups=False))
tab.to_csv(f"{OUT}/table_main.csv"); tab
""")

md(r"""
## 4. RQ2 — Seberapa besar metrik tergelembung jika split tidak benar?
Model yang sama (H = 24 jam) dievaluasi dengan tiga protokol: **acak** (baris diacak, kolam & waktu bocor), **per kolam** (waktu bocor), dan **per kolam + waktu** (protokol utama).
""")
code(r"""
rq2 = [res[(res.horizon == 24) & (res.model != "rule")]]
rq2.append(evaluate(X, Y, meta, 24, models=("logreg", "rf", "lgbm"), protocol="random"))
rq2.append(evaluate(X, Y, meta, 24, models=("logreg", "rf", "lgbm"), protocol="pond", n_splits=N_SPLITS))
rq2 = pd.concat(rq2); rq2.to_csv(f"{OUT}/results_rq2.csv", index=False)
piv = rq2.groupby(["model", "protocol"]).pr_auc.mean().unstack()[["random", "pond", "pond_time"]]
piv["inflasi_random_vs_pond_time_%"] = 100 * (piv["random"] / piv["pond_time"] - 1)
piv.round(3)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
names = {"logreg": "Logistic Reg.", "rf": "Random Forest", "lgbm": "LightGBM", "rule": "Aturan DO_min24"}
cols_p = {"random": "#e53e3e", "pond": "#dd6b20", "pond_time": "#2b6cb0"}
lab_p = {"random": "split acak (bocor)", "pond": "split per kolam", "pond_time": "split per kolam + waktu"}
x = np.arange(3); w = 0.26
for i, p in enumerate(["random", "pond", "pond_time"]):
    v = piv[p].reindex(["logreg", "rf", "lgbm"])
    ax[0].bar(x + (i - 1) * w, v, w, color=cols_p[p], label=lab_p[p])
    for xi, vi in zip(x + (i - 1) * w, v): ax[0].text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=8)
ax[0].set_xticks(x, [names[m] for m in ["logreg", "rf", "lgbm"]]); ax[0].set_ylabel("PR-AUC (H = 24 jam)"); ax[0].set_ylim(0, 1)
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) Inflasi metrik akibat protokol split", loc="left")

pr = res.groupby(["horizon", "model"]).pr_auc.agg(["mean", "std"]).reset_index()
for m, col in zip(["rule", "logreg", "rf", "lgbm"], ["#a0aec0", "#68d391", "#dd6b20", "#2b6cb0"]):
    s = pr[pr.model == m]
    ax[1].errorbar(s.horizon, s["mean"], yerr=s["std"], marker="o", capsize=3, color=col, label=names[m])
prev = res.groupby("horizon").prevalence.mean()
ax[1].plot(prev.index, prev.values, "k:", label="tebakan acak (prevalensi)")
ax[1].set_xticks(list(horizons)); ax[1].set_xlabel("horizon prediksi (jam)"); ax[1].set_ylabel("PR-AUC (split per kolam + waktu)")
ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("(b) Performa per horizon", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_models.png", dpi=160); plt.show()
""")

md(r"""
## 5. Contoh early warning pada kolam yang tidak pernah dilihat model
Model LightGBM (H = 24 jam) dilatih pada kolam fold-0 periode Jan–Apr, lalu dijalankan seperti di perangkat pada kolam uji periode Mei–Jun.
""")
code(r"""
from sklearn.model_selection import GroupKFold
H = 24; y = np.nan_to_num(Y[f"y{H}"].values).astype(int)
ok = Y[f"y{H}"].notna().values & X["DO_lag24"].notna().values
idx = np.where(ok)[0]
trp, tep = next(GroupKFold(5).split(idx, groups=meta.pond_id.values[idx]))
tr, te = idx[trp], idx[tep]
tr_in = tr[meta.datetime.values[tr] < CUT_VAL]; va = tr[(meta.datetime.values[tr] >= CUT_VAL) & (meta.datetime.values[tr] < CUT_TEST)]
rng = np.random.default_rng(0)
s_in = downsample(tr_in, y, rng)
m_in = make_model("lgbm").fit(X.iloc[s_in], y[s_in])
thr = best_f1_threshold(y[va], m_in.predict_proba(X.iloc[va])[:, 1])
tr_full = tr[meta.datetime.values[tr] < CUT_TEST]; sub = downsample(tr_full, y, rng)
model = make_model("lgbm").fit(X.iloc[sub], y[sub])
te = te[meta.datetime.values[te] >= CUT_TEST]
score = pd.Series(model.predict_proba(X.iloc[te])[:, 1], index=te)
print(f"threshold alarm = {thr:.3f}")

tm = meta.loc[te].assign(score=score.values)
pid = tm.groupby("pond_id").deaths.sum().idxmax()
p = tm[tm.pond_id == pid]
worst = p.set_index("datetime").deaths.resample("D").sum().idxmax()
w = p[(p.datetime >= worst - pd.Timedelta(days=5)) & (p.datetime <= worst + pd.Timedelta(days=2))]
wd = d.loc[w.index]
fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[1.2, 1])
ax[0].plot(wd.datetime, wd.sensor_do_mgl, color="#2b6cb0", lw=1.2, label="DO terbaca sensor")
ax[0].set_ylabel("DO (mg/L)"); a2 = ax[0].twinx(); a2.bar(w.datetime, w.deaths, width=0.04, color="#e53e3e", alpha=0.7, label="ikan mati / jam")
a2.set_ylabel("ikan mati / jam", color="#e53e3e"); a2.spines["top"].set_visible(False)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
ax[0].set_title(f"Kolam uji {pid}: skor risiko model vs kejadian kematian", loc="left")
ax[1].plot(w.datetime, w.score, color="#553c9a", lw=1.4, label="skor risiko (LightGBM, H=24 j)")
ax[1].axhline(thr, ls="--", color="#e53e3e", lw=0.9, label="threshold alarm")
ax[1].fill_between(w.datetime, 0, 1, where=w.score >= thr, color="#e53e3e", alpha=0.12, transform=ax[1].get_xaxis_transform(), label="ALARM")
ax[1].set_ylabel("probabilitas"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_early_warning_example.png", dpi=160); plt.show()
""")

md("## 6. Fitur paling berpengaruh (gain LightGBM) — analisis SHAP lengkap di Bagian 3")
code(r"""
imp = pd.Series(model.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
imp = 100 * imp / imp.sum()
fig, ax = plt.subplots(figsize=(7, 4.5))
imp.head(15)[::-1].plot.barh(ax=ax, color="#2b6cb0"); ax.set_xlabel("% total gain")
ax.set_title("15 fitur teratas (LightGBM, H = 24 jam)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_feature_importance.png", dpi=160); plt.show()
imp.head(15).round(2)
""")

md("## 7. Simpan ringkasan")
code(r"""
summary_json = {
    "prevalence": Y.mean().round(4).to_dict(),
    "main_pr_auc": res.groupby(["horizon", "model"]).pr_auc.mean().round(3).unstack().to_dict(),
    "rq2_pr_auc_h24": piv.round(3).to_dict(),
    "top_features": imp.head(10).round(2).to_dict(),
}
json.dump(summary_json, open(f"{OUT}/part2_summary.json", "w"), indent=1, default=str)
summary_json
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, sys.argv[1]); print("cells:", len(cells))
