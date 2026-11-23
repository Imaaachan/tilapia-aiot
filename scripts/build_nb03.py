"""Build notebooks/03_analysis.ipynb. Run from the repository root."""

from pathlib import Path

import nbformat as nbf


cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(
    r"""
# Bagian 3 — Interpretabilitas, ablation sensor, dan model sequence

Analisis lanjutan memakai digital twin dan label yang sama dengan notebook 02.
Semua perbandingan utama mengikuti split **pond_time**: kolam uji tidak terlihat
saat training dan periode uji hanya Mei–Juni. Data adalah simulasi, bukan
pengamatan lapangan.

Isi: SHAP untuk LightGBM H=6/24, ablation kelompok sensor H=6/12/24, CNN 1D
sebagai model sequence pembanding, serta uji Wilcoxon/Friedman pada hasil
5-fold notebook 02. CNN memakai 5 fold; bila waktu CPU melebihi 30 menit,
turunkan menjadi 3 fold dan laporkan.
"""
)

code(
    r"""
import json, os, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Jalankan notebook dari repo local yang berisi src/.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

from features import clean, make_features, make_labels
from modeling import (
    CUT_TEST, CUT_VAL, best_f1_threshold, downsample, event_metrics, make_model,
)

SEED = 0
N_SPLITS = 5
CNN_SPLITS = 5
CNN_EPOCHS = 3
CNN_BATCH_SIZE = 1024
torch.set_num_threads(min(4, os.cpu_count() or 1))
DEVICE = torch.device("cpu")
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
print(f"Repo: {ROOT}")
print(f"PyTorch {torch.__version__}; device: {DEVICE}; SHAP {shap.__version__}")
"""
)

md(
    r"""
## 1. Data dan fold pond_time

Muat dataset sintetis yang dibuat notebook 01. Fitur berasal dari `clean()` dan
`make_features()`; kolom target dan state internal simulator hanya dipakai
untuk label/metadata evaluasi, bukan sebagai fitur.
"""
)

code(
    r"""
SIM = OUT / "sim_ponds.csv.gz"
if not SIM.exists():
    raise FileNotFoundError(
        "Dataset simulasi tidak ditemukan. Jalankan notebook 01 terlebih dahulu."
    )
d = pd.read_csv(SIM, parse_dates=["datetime"])
d = d.sort_values(["pond_id", "datetime"]).reset_index(drop=True)
c = clean(d)
X = make_features(c)
Y = make_labels(d)
meta = d[["pond_id", "datetime", "alive", "deaths", "true_do_mgl"]].copy()
print(f"Data: {d.shape}; kolam: {d.pond_id.nunique()}; fitur: {X.shape[1]}")
print("Prevalensi:", Y.mean().round(4).to_dict())
"""
)

code(
    r"""
def pond_time_folds(horizon, n_splits=N_SPLITS, require_72h=False):
    y_all = Y[f"y{horizon}"].to_numpy()
    valid = Y[f"y{horizon}"].notna().to_numpy() & X["DO_lag24"].notna().to_numpy()
    if require_72h:
        valid &= d.groupby("pond_id", sort=False).cumcount().to_numpy() >= 71
    idx = np.flatnonzero(valid)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_pos, test_pos) in enumerate(
        splitter.split(idx, groups=meta.pond_id.to_numpy()[idx])
    ):
        train = idx[train_pos]
        test = idx[test_pos]
        train = train[meta.datetime.to_numpy()[train] < CUT_TEST]
        test = test[meta.datetime.to_numpy()[test] >= CUT_TEST]
        inner = train[meta.datetime.to_numpy()[train] < CUT_VAL]
        validation = train[
            (meta.datetime.to_numpy()[train] >= CUT_VAL)
            & (meta.datetime.to_numpy()[train] < CUT_TEST)
        ]
        yield fold, train, inner, validation, test


def train_lgbm(features, y, indices, rng):
    sampled = downsample(indices, y, rng)
    model = make_model("lgbm", seed=SEED)
    model.fit(features.iloc[sampled], y[sampled])
    return model


def evaluate_lgbm_features(features, horizon):
    y = np.nan_to_num(Y[f"y{horizon}"].to_numpy()).astype(int)
    rng = np.random.default_rng(SEED)
    rows = []
    for fold, train, inner, validation, test in pond_time_folds(horizon):
        model = train_lgbm(features, y, train, rng)
        scores = model.predict_proba(features.iloc[test])[:, 1]
        inner_model = train_lgbm(features, y, inner, rng)
        val_scores = inner_model.predict_proba(features.iloc[validation])[:, 1]
        threshold = best_f1_threshold(y[validation], val_scores)
        alarm = (scores >= threshold).astype(int)
        rows.append(
            {
                "horizon": horizon,
                "fold": fold,
                "pr_auc": average_precision_score(y[test], scores),
                "threshold": float(threshold),
                **event_metrics(meta.iloc[test].reset_index(drop=True), y[test], alarm),
            }
        )
    return pd.DataFrame(rows)
"""
)

md(
    r"""
## 2. SHAP — LightGBM H=6 dan H=24

Untuk tiap horizon, model fold-0 dilatih pada kolam training Jan–Apr. Plot
memakai sampel tetap dari kolam uji Mei–Juni. Ketergantungan DO ditampilkan
dengan ambang biologis 0,8, 2, dan 4 mg/L sebagai acuan, bukan sebagai ambang
yang dipelajari model. Titik perubahan empiris dihitung dari perubahan terbesar
antar-bin SHAP dan disajikan sebagai deskriptif.
"""
)

code(
    r"""
def fit_shap_fold0(horizon):
    y = np.nan_to_num(Y[f"y{horizon}"].to_numpy()).astype(int)
    fold, train, inner, validation, test = next(pond_time_folds(horizon))
    rng = np.random.default_rng(SEED)
    # Urutan sampling meniru contoh fold-0 notebook 02: threshold dahulu, lalu model penuh.
    inner_sample = downsample(inner, y, rng)
    inner_model = make_model("lgbm", seed=SEED).fit(X.iloc[inner_sample], y[inner_sample])
    threshold = best_f1_threshold(
        y[validation], inner_model.predict_proba(X.iloc[validation])[:, 1]
    )
    full_sample = downsample(train, y, rng)
    model = make_model("lgbm", seed=SEED).fit(X.iloc[full_sample], y[full_sample])
    sample_idx = np.random.default_rng(SEED + horizon).choice(
        test, size=min(2500, len(test)), replace=False
    )
    sample = X.iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return model, sample, values, threshold


def empirical_turning_point(feature, sample, values):
    x = sample[feature].to_numpy(dtype=float)
    s = values[:, X.columns.get_loc(feature)].astype(float)
    edges = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0, 1, 21)))
    if len(edges) < 4:
        return {"feature": feature, "turning_point_mgl": None, "bins": []}
    ids = np.digitize(x, edges[1:-1], right=True)
    grouped = []
    for bin_id in np.unique(ids):
        selected = (ids == bin_id) & np.isfinite(x) & np.isfinite(s)
        if selected.sum():
            grouped.append(
                {
                    "center": float(np.mean(x[selected])),
                    "mean_shap": float(np.mean(s[selected])),
                }
            )
    if len(grouped) < 3:
        point = None
    else:
        centers = np.array([item["center"] for item in grouped])
        means = np.array([item["mean_shap"] for item in grouped])
        slopes = np.abs(np.diff(means) / np.maximum(np.diff(centers), 1e-8))
        point = float((centers[np.argmax(slopes)] + centers[np.argmax(slopes) + 1]) / 2)
    return {"feature": feature, "turning_point_mgl": point, "bins": grouped}


shap_results = {}
for horizon in (6, 24):
    model, sample, values, threshold = fit_shap_fold0(horizon)
    mean_abs = np.abs(values).mean(axis=0)
    top_order = np.argsort(mean_abs)[::-1]
    top_features = X.columns[top_order[:10]].tolist()

    shap.summary_plot(values, sample, max_display=15, show=False, plot_size=(10, 6))
    plt.title(f"LightGBM H={horizon} jam — SHAP beeswarm (fold 0)")
    plt.tight_layout()
    plt.savefig(OUT / f"fig_shap_summary_h{horizon}.png", dpi=160, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    dependence_features = top_features[:4]
    for ax, feature in zip(axes.flat, dependence_features):
        col = X.columns.get_loc(feature)
        x = sample[feature].to_numpy()
        color = values[:, col]
        points = ax.scatter(x, values[:, col], c=color, cmap="coolwarm", s=8, alpha=0.45)
        if feature in ("DO_min24", "DO_min72"):
            for value, label in ((0.8, "TLM 0.8"), (2.0, "kritis 2"), (4.0, "stres 4")):
                ax.axvline(value, color="black", linestyle="--", linewidth=0.8, alpha=0.65)
                ax.text(value, 0.98, label, rotation=90, va="top", ha="right",
                        transform=ax.get_xaxis_transform(), fontsize=7)
        ax.set(title=feature, xlabel="Nilai fitur", ylabel="SHAP value")
        fig.colorbar(points, ax=ax, label="SHAP value", fraction=0.046, pad=0.04)
    fig.suptitle(f"Dependence plot — LightGBM H={horizon} jam, fold 0")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_shap_dependence_h{horizon}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    turning = [
        empirical_turning_point(feature, sample, values)
        for feature in ("DO_min24", "DO_min72")
    ]
    shap_results[str(horizon)] = {
        "threshold_h24": float(threshold),
        "top_features_by_mean_abs_shap": [
            {"feature": X.columns[i], "mean_abs_shap": float(mean_abs[i])}
            for i in top_order[:10]
        ],
        "do_dependence_bins_and_empirical_turning_points": turning,
        "biological_reference_thresholds_mgl": {"effective_tlm": 0.8, "very_low_oxygen": 2.0, "stress": 4.0},
    }
    print(f"H={horizon}: top features {top_features[:4]}; DO turning points {[(r['feature'], r['turning_point_mgl']) for r in turning]}")
"""
)

md(
    r"""
## 3. Ablation kelompok sensor

Bandingkan semua fitur, DO saja (+ aerator/jam), DO+suhu, dan tanpa DO
(suhu+pH+turbiditas, aerator/jam tetap tersedia). Fitur saturasi oksigen
terkoreksi suhu hanya dimasukkan pada konfigurasi yang memiliki sensor suhu.
`miss6` tidak disertakan pada konfigurasi terablasi karena mencampur status
hilang dari semua sensor.
"""
)

code(
    r"""
hour_features = ["hour_sin", "hour_cos"]
aerator_features = ["aer_on", "aer_frac24"]
do_features = [
    name for name in X.columns
    if (name == "DO" or name.startswith("DO_") or name.startswith("hrs_DO_"))
    and not name.endswith("_pct_sat")
]
temperature_features = [
    name for name in X.columns if name == "T" or name.startswith("T_")
]
ph_features = [name for name in X.columns if name == "pH" or name.startswith("pH_")]
turbidity_features = [
    name for name in X.columns if name == "turb" or name.startswith("turb_")
]

ablation_features = {
    "all_sensors": X.columns.tolist(),
    "DO_only": hour_features + aerator_features + do_features,
    "DO_plus_temperature": (
        hour_features + aerator_features + do_features + temperature_features
        + ["DO_pct_sat", "DO_min24_pct_sat"]
    ),
    "no_DO": hour_features + aerator_features + temperature_features
    + ph_features + turbidity_features,
}
for name, selected in ablation_features.items():
    assert selected and len(selected) == len(set(selected)), name
    print(name, len(selected), selected[:5])

ablation_folds = []
for name, selected in ablation_features.items():
    for horizon in (6, 12, 24):
        started = time.time()
        fold_results = evaluate_lgbm_features(X[selected], horizon)
        fold_results["ablation"] = name
        ablation_folds.append(fold_results)
        print(f"{name}, H={horizon}: {time.time() - started:.0f} s")
ablation_folds = pd.concat(ablation_folds, ignore_index=True)
ablation_folds.to_csv(OUT / "ablation_sensor_folds.csv", index=False)
ablation_summary = (
    ablation_folds.groupby(["horizon", "ablation"])
    .agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
        event_recall_mean=("event_recall", "mean"),
        event_recall_sd=("event_recall", "std"),
    )
    .reset_index()
)
ablation_summary.to_csv(OUT / "ablation_sensor_summary.csv", index=False)
ablation_summary.round(3)
"""
)

code(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
labels = {
    "all_sensors": "Semua",
    "DO_only": "DO saja",
    "DO_plus_temperature": "DO + suhu",
    "no_DO": "Tanpa DO",
}
colors = dict(zip(ablation_features, ("#2b6cb0", "#dd6b20", "#38a169", "#805ad5")))
for ax, metric, ylabel in (
    (axes[0], "pr_auc_mean", "PR-AUC"),
    (axes[1], "event_recall_mean", "Event recall"),
):
    for ablation, group in ablation_summary.groupby("ablation"):
        group = group.sort_values("horizon")
        err_col = metric.replace("_mean", "_sd")
        ax.errorbar(
            group["horizon"], group[metric], yerr=group[err_col],
            marker="o", capsize=3, label=labels[ablation], color=colors[ablation],
        )
    ax.set(xticks=[6, 12, 24], xlabel="Horizon (jam)", ylabel=ylabel, ylim=(0, 1))
    ax.legend(frameon=False)
fig.suptitle("Ablation kelompok sensor — pond_time, 5 fold")
fig.tight_layout()
fig.savefig(OUT / "fig_ablation_sensors.png", dpi=160, bbox_inches="tight")
plt.close(fig)
"""
)

md(
    r"""
## 4. Model sequence: CNN 1D kecil (CPU)

Masukan tiap contoh adalah jendela 72 jam dari empat bacaan sensor, status
aerator, dan sin/cos jam. Nilai sensor dinormalisasi memakai statistik periode
training fold saja; nilai yang masih hilang setelah preprocessing diisi median
training. Negatif di-*downsample* dengan helper yang sama (`neg_frac=0.25`).
CNN memakai 5 fold dan 3 epoch. Threshold tetap
ditentukan dari April kolam training, lalu hasil uji di Mei–Juni.
Model ini baseline kecil (3 epoch), bukan hasil tuning arsitektur; metriknya
dilaporkan apa adanya dan tidak dipakai untuk menggantikan model utama.
"""
)

code(
    r"""
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

hour = d.datetime.dt.hour.to_numpy()
sequence_raw = np.column_stack(
    [
        c[["T", "DO", "pH", "turb"]].to_numpy(dtype=np.float32),
        c["aerator_on"].to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * hour / 24).astype(np.float32),
        np.cos(2 * np.pi * hour / 24).astype(np.float32),
    ]
).astype(np.float32)
local_hour = d.groupby("pond_id", sort=False).cumcount().to_numpy()
WINDOW = 72
OFFSETS = np.arange(WINDOW - 1, -1, -1)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, sequence):
        return self.net(sequence).squeeze(1)


def fit_sequence_scaler(train_idx):
    sensor = sequence_raw[train_idx, :4]
    median = np.nanmedian(sensor, axis=0)
    median = np.where(np.isfinite(median), median, 0).astype(np.float32)
    filled = np.where(np.isfinite(sensor), sensor, median)
    mean = filled.mean(axis=0).astype(np.float32)
    scale = filled.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1).astype(np.float32)
    return median, mean, scale


def make_windows(indices, scaler):
    median, mean, scale = scaler
    rows = indices[:, None] - OFFSETS[None, :]
    windows = sequence_raw[rows].copy()
    sensors = windows[:, :, :4]
    sensors = np.where(np.isfinite(sensors), sensors, median[None, None, :])
    windows[:, :, :4] = (sensors - mean[None, None, :]) / scale[None, None, :]
    return torch.from_numpy(windows.transpose(0, 2, 1).copy()).to(DEVICE)


def fit_cnn(train_idx, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampled = downsample(train_idx, y, rng, neg_frac=0.25)
    scaler = fit_sequence_scaler(train_idx)
    training = TensorDataset(
        make_windows(sampled, scaler),
        torch.from_numpy(y[sampled].astype(np.float32)),
    )
    loader = DataLoader(training, batch_size=CNN_BATCH_SIZE, shuffle=True)
    model = SmallCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=DEVICE))
    model.train()
    for _ in range(CNN_EPOCHS):
        for sequences, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), targets)
            loss.backward()
            optimizer.step()
    return model, scaler


def predict_cnn(model, scaler, indices):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), 2048):
            batch_idx = indices[start : start + 2048]
            logits = model(make_windows(batch_idx, scaler))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions)


def evaluate_cnn(horizon, n_splits=CNN_SPLITS):
    y = np.nan_to_num(Y[f"y{horizon}"].to_numpy()).astype(int)
    rows = []
    for fold, train, inner, validation, test in pond_time_folds(
        horizon, n_splits=n_splits, require_72h=True
    ):
        started = time.time()
        validation_model, validation_scaler = fit_cnn(inner, y, SEED + fold)
        val_scores = predict_cnn(validation_model, validation_scaler, validation)
        threshold = best_f1_threshold(y[validation], val_scores)
        del validation_model

        model, scaler = fit_cnn(train, y, SEED + 100 + fold)
        scores = predict_cnn(model, scaler, test)
        alarm = (scores >= threshold).astype(int)
        rows.append(
            {
                "horizon": horizon,
                "fold": fold,
                "pr_auc": average_precision_score(y[test], scores),
                "threshold": float(threshold),
                **event_metrics(meta.iloc[test].reset_index(drop=True), y[test], alarm),
                "fit_and_score_s": time.time() - started,
                "epochs": CNN_EPOCHS,
            }
        )
        del model
        print(f"CNN H={horizon}, fold={fold}: {time.time() - started:.0f} s")
    return pd.DataFrame(rows)


sequence_rows = []
for horizon in (6, 12, 24):
    sequence_rows.append(evaluate_cnn(horizon))
sequence_results = pd.concat(sequence_rows, ignore_index=True)
sequence_results.to_csv(OUT / "sequence_cnn_folds.csv", index=False)

main_results = pd.read_csv(OUT / "results_pond_time.csv")
lgbm_matched = main_results[
    (main_results["protocol"] == "pond_time")
    & (main_results["model"] == "lgbm")
    & (main_results["fold"] < CNN_SPLITS)
][["horizon", "fold", "pr_auc", "event_recall"]].rename(
    columns={"pr_auc": "lgbm_pr_auc", "event_recall": "lgbm_event_recall"}
)
sequence_comparison = sequence_results.merge(
    lgbm_matched, on=["horizon", "fold"], validate="one_to_one"
)
sequence_comparison.to_csv(OUT / "sequence_cnn_vs_lgbm.csv", index=False)
sequence_comparison.groupby("horizon")[
    ["pr_auc", "lgbm_pr_auc", "event_recall", "lgbm_event_recall"]
].mean().round(3)
"""
)

md(
    r"""
## 5. Uji statistik hasil utama

Gunakan hasil 5-fold × 3 horizon yang sudah tersimpan dari notebook 02.
Pasangan dibentuk berdasarkan horizon dan fold (15 pasangan). Wilcoxon
signed-rank membandingkan LightGBM dengan RF dan Logistic Regression;
Friedman membandingkan ketiga model secara berpasangan pada PR-AUC.
Karena horizon pada fold yang sama berkorelasi, p-value ini bersifat
eksploratif dan bukan 15 sampel independen.
"""
)

code(
    r"""
paired = main_results[main_results["protocol"] == "pond_time"].pivot(
    index=["horizon", "fold"], columns="model", values="pr_auc"
).dropna(subset=["lgbm", "rf", "logreg"])
if len(paired) != 15:
    raise ValueError(f"Diharapkan 15 pasangan fold-horizon, ditemukan {len(paired)}.")

stat_tests = {}
for other in ("rf", "logreg"):
    result = wilcoxon(paired["lgbm"], paired[other], alternative="two-sided")
    stat_tests[f"wilcoxon_lgbm_vs_{other}"] = {
        "n_pairs": int(len(paired)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_paired_difference_lgbm_minus_other": float(
            (paired["lgbm"] - paired[other]).median()
        ),
    }
friedman = friedmanchisquare(paired["lgbm"], paired["rf"], paired["logreg"])
stat_tests["friedman_3_models"] = {
    "n_blocks": int(len(paired)),
    "statistic": float(friedman.statistic),
    "p_value": float(friedman.pvalue),
}
stat_tests
"""
)

md(
    r"""
## 6. Simpan tabel dan ringkasan hasil

Ringkasan memuat angka yang terukur, konfigurasi run, ambang biologis rujukan,
serta catatan bahwa data berasal dari simulator. Gambar SHAP, ablation, dan
metrik per-fold disimpan di `outputs/`.
"""
)

code(
    r"""
ablation_json = {}
for horizon, group in ablation_summary.groupby("horizon"):
    ablation_json[str(int(horizon))] = {
        row["ablation"]: {
            "pr_auc_mean": float(row["pr_auc_mean"]),
            "pr_auc_sd": None if pd.isna(row["pr_auc_sd"]) else float(row["pr_auc_sd"]),
            "event_recall_mean": float(row["event_recall_mean"]),
            "event_recall_sd": None if pd.isna(row["event_recall_sd"]) else float(row["event_recall_sd"]),
        }
        for _, row in group.iterrows()
    }

sequence_json = {}
for horizon, group in sequence_comparison.groupby("horizon"):
    sequence_json[str(int(horizon))] = {
        "n_folds": int(group["fold"].nunique()),
        "cnn_pr_auc_mean": float(group["pr_auc"].mean()),
        "cnn_pr_auc_sd": float(group["pr_auc"].std()),
        "lgbm_matched_pr_auc_mean": float(group["lgbm_pr_auc"].mean()),
        "cnn_event_recall_mean": float(group["event_recall"].mean()),
        "lgbm_matched_event_recall_mean": float(group["lgbm_event_recall"].mean()),
        "cnn_epochs": CNN_EPOCHS,
    }

summary = {
    "dataset": "Synthetic tilapia pond digital twin; not field observations",
    "evaluation_protocol": "pond_time: GroupKFold by pond; train Jan-Apr, test May-Jun; threshold tuned on April",
    "shap": shap_results,
    "sensor_ablation": ablation_json,
    "sequence_cnn": {
        "architecture": "two-layer 1D CNN, 7 input channels, global average pooling",
        "folds": CNN_SPLITS,
        "sequence_hours": WINDOW,
        "negative_sampling_fraction": 0.25,
        "results": sequence_json,
    },
    "statistical_tests": {
        "metric": "PR-AUC",
        "paired_design": "5 folds x 3 horizons = 15 paired fold-horizon observations",
        **stat_tests,
    },
}
summary_path = OUT / "part3_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Ringkasan tersimpan: {summary_path}")
print(json.dumps(summary["statistical_tests"], indent=2))
"""
)

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "tilapia-aiot",
        },
        "language_info": {"name": "python"},
    },
)
target = Path("notebooks/03_analysis.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(f"Created {target}")
