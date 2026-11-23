"""Simulator validation figures + summary. Usage: python3 -I scripts/validate_sim.py sim/sim_ponds.parquet out/"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = pd.read_parquet(src)
d["m"] = d.datetime.dt.month; d["h"] = d.datetime.dt.hour
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
ref = {"T": [27.47, 26.50, 28.35, 27.70, 26.60, 27.45], "pH": [7.98, 7.34, 7.97, 7.91, 7.89, 7.88],
       "surv": [95.27, 93.05, 96.30, 92.96, 92.93, 92.85]}

g = d.groupby(["pond_id", "m"]).agg(start=("alive", "first"), dead=("deaths", "sum"))
g["surv"] = 100 * (1 - g.dead / g.start)
summary = {
    "rows": len(d), "ponds": int(d.pond_id.nunique()),
    "sim_T_month": d.groupby("m").true_temp_c.mean().round(2).tolist(), "ref_T_month": ref["T"],
    "sim_pH_month": d.groupby("m").true_ph.mean().round(2).tolist(), "ref_pH_month": ref["pH"],
    "sim_surv_month_mean": g.surv.groupby("m").mean().round(2).tolist(),
    "sim_surv_month_median": g.surv.groupby("m").median().round(2).tolist(), "ref_surv_month": ref["surv"],
    "sim_surv_overall_mean": round(g.surv.mean(), 2), "ref_surv_overall_mean": round(np.mean(ref["surv"]), 2),
    "pond_months_below_90pct": round((g.surv < 90).mean(), 3),
    "DO_hour_min": int(d.groupby("h").true_do_mgl.mean().idxmin()), "DO_hour_max": int(d.groupby("h").true_do_mgl.mean().idxmax()),
    "DO_dawn_mean": round(d[d.h == 5].true_do_mgl.mean(), 2), "DO_15h_mean": round(d[d.h == 15].true_do_mgl.mean(), 2),
    "corr_T_true_vs_sensor": round(d[["true_temp_c", "sensor_temp_c"]].corr().iloc[0, 1], 4),
    "MAE_DO_sensor": round((d.sensor_do_mgl - d.true_do_mgl).abs().mean(), 3),
    "MAE_pH_sensor": round((d.sensor_ph - d.true_ph).abs().mean(), 3),
}
json.dump(summary, open(f"{out}/sim_validation.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.15, 1])

ax = fig.add_subplot(gs[0, 0])
q = d.groupby("h").true_do_mgl.quantile([0.1, 0.5, 0.9]).unstack()
ax.fill_between(q.index, q[0.1], q[0.9], color="#90cdf4", alpha=0.6, label="P10–P90 (200 kolam)")
ax.plot(q.index, q[0.5], color="#2b6cb0", lw=2, label="median")
ax.axhline(4, ls="--", color="#dd6b20", lw=0.9); ax.text(0.3, 4.15, "batas stres 4 mg/L", color="#dd6b20", fontsize=8)
ax.set_xlabel("jam"); ax.set_ylabel("DO (mg/L)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(a) Siklus harian DO: minimum menjelang fajar, maksimum sore", loc="left")

ax = fig.add_subplot(gs[0, 1])
x = np.arange(6)
ax.plot(x, ref["T"], "o-", color="#e53e3e", label="Montería (rata-rata bulanan)")
ax.plot(x, summary["sim_T_month"], "s--", color="#2b6cb0", label="simulasi")
ax.set_xticks(x, MON); ax.set_ylabel("suhu air (°C)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(b) Kalibrasi suhu bulanan terhadap Montería", loc="left")

ax = fig.add_subplot(gs[1, :])
ev = d.groupby("pond_id").deaths.sum().sort_values()
pid = ev.index[int(len(ev) * 0.9)]
p = d[(d.pond_id == pid)]
day_dead = p.set_index("datetime").deaths.resample("D").sum()
worst = day_dead.idxmax()
w = p[(p.datetime >= worst - pd.Timedelta(days=6)) & (p.datetime <= worst + pd.Timedelta(days=3))]
ax.plot(w.datetime, w.true_do_mgl, color="#2b6cb0", lw=1.5, label="DO aktual")
ax.plot(w.datetime, w.sensor_do_mgl, color="#a0aec0", lw=0.8, label="DO terbaca sensor (noise/drift/missing)")
ax.axhline(0.8, ls="--", color="#e53e3e", lw=0.8); ax.set_ylabel("DO (mg/L)")
ax2 = ax.twinx(); ax2.bar(w.datetime, w.deaths, width=0.04, color="#e53e3e", alpha=0.7, label="ikan mati / jam")
ax2.set_ylabel("ikan mati / jam", color="#e53e3e"); ax2.spines["top"].set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"(c) Contoh kejadian: kolam {pid} — DO turun beberapa malam berturut-turut sebelum kematian massal (sinyal early warning)", loc="left")

ax = fig.add_subplot(gs[2, :])
data = [g.xs(m, level="m").surv.clip(lower=60) for m in range(1, 7)]
ax.boxplot(data, positions=x, widths=0.5, showfliers=True, flierprops=dict(markersize=2, alpha=0.4))
ax.plot(x, ref["surv"], "o", color="#e53e3e", ms=7, label="Montería")
ax.plot(x, summary["sim_surv_month_mean"], "s", color="#2b6cb0", ms=6, label="rata-rata simulasi")
ax.set_xticks(x, MON); ax.set_ylabel("survival bulanan (%)"); ax.set_ylim(59, 100); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("(d) Distribusi survival bulanan per kolam (nilai <60% dipotong ke 60) vs Montería", loc="left")

fig.tight_layout(); fig.savefig(f"{out}/fig_sim_validation.png", dpi=160)
