"""Leakage-safe evaluation of early-warning models.

Protocol (main): 5-fold GroupKFold over ponds  x  temporal cut
   train = training ponds, Jan–Apr   (threshold tuned on April of training ponds, model fit Jan–Mar for tuning)
   test  = unseen ponds,   May–Jun
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
import lightgbm as lgb

CUT_TEST = pd.Timestamp("2024-05-01")
CUT_VAL = pd.Timestamp("2024-04-01")


def rule_score(X):
    return -X["DO_min24"].fillna(X["DO_min24"].median()).values   # lower overnight DO -> higher risk


def make_model(name, seed=0):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=300, C=0.5))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, max_samples=0.3,
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    if name == "lgbm":
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=10,
                                  random_state=seed, verbose=-1, n_jobs=-1)
    raise ValueError(name)


def downsample(idx, y, rng, neg_frac=0.25):
    pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    return np.sort(np.concatenate([pos, rng.choice(neg, int(len(neg) * neg_frac), replace=False)]))


def best_f1_threshold(y, s):
    order = np.argsort(-s); ys = y[order]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / max(ys.sum(), 1); f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1)); return s[order][i]


def fit_score(name, X, y, tr, te, rng):
    if name == "rule":
        return rule_score(X.iloc[te])
    sub = downsample(tr, y, rng)
    m = make_model(name).fit(X.iloc[sub], y[sub])
    return m.predict_proba(X.iloc[te])[:, 1]


def event_metrics(meta_te, y_te, alarm, lookback=48):
    """Event = run of consecutive positive labels. Onset = first hour >= run start with deaths >= 5/10k fish.
    Detected if an alarm fires in [onset-lookback, onset). Lead = onset - first such alarm (h)."""
    res = []; fa_hours = 0
    for pid, grp in meta_te.assign(y=y_te, a=alarm).groupby("pond_id"):
        yv = grp.y.values; av = grp.a.values; dv = grp.deaths.values; al = grp.alive.values
        starts = np.where((yv == 1) & (np.r_[0, yv[:-1]] != 1))[0]
        covered = np.zeros(len(yv), bool)
        for s in starts:
            thr = np.maximum(5e-4 * np.r_[al[0], al[:-1]], 1)
            on = s + np.argmax(dv[s:] >= thr[s:]) if (dv[s:] >= thr[s:]).any() else min(s + 24, len(yv) - 1)
            lo = max(0, on - lookback)
            covered[lo:on + 48] = True
            w = np.where(av[lo:on] == 1)[0]
            res.append({"detected": len(w) > 0, "lead_h": on - (lo + w[0]) if len(w) else np.nan})
        fa_hours += int(((av == 1) & ~covered).sum())
    ev = pd.DataFrame(res)
    n_pm = meta_te.groupby(["pond_id", meta_te.datetime.dt.month]).ngroups
    # false-alarm episodes per pond-month (alarm runs outside any event window)
    return {"events": len(ev), "event_recall": ev.detected.mean() if len(ev) else np.nan,
            "median_lead_h": ev.lead_h.median() if len(ev) else np.nan,
            "false_alarm_h_per_pond_month": fa_hours / n_pm}


def evaluate(X, Y, meta, horizon, models=("rule", "logreg", "rf", "lgbm"), protocol="pond_time", n_splits=5, seed=0):
    y_all = Y[f"y{horizon}"].values
    ok = ~np.isnan(y_all) & X["DO_lag24"].notna().values
    rng = np.random.default_rng(seed)
    rows = []
    if protocol == "random":
        idx = np.where(ok)[0]
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y_all[idx])
        folds = [(tr, te, None)]
    else:
        folds = []
        gkf = GroupKFold(n_splits=n_splits)
        idx_all = np.where(ok)[0]
        for f, (trp, tep) in enumerate(gkf.split(idx_all, groups=meta.pond_id.values[idx_all])):
            tr, te = idx_all[trp], idx_all[tep]
            if protocol == "pond_time":
                tr = tr[meta.datetime.values[tr] < CUT_TEST]; te = te[meta.datetime.values[te] >= CUT_TEST]
            folds.append((tr, te, tr[meta.datetime.values[tr] < CUT_VAL]))
    y = np.nan_to_num(y_all).astype(int)
    for f, (tr, te, tr_inner) in enumerate(folds):
        for name in models:
            t0 = time.time()
            s = fit_score(name, X, y, tr, te, rng)
            # threshold: tuned on April of training ponds with a model fit on Jan–Mar (no test information)
            if tr_inner is not None:
                va = np.setdiff1d(tr, tr_inner)
                sv = fit_score(name, X, y, tr_inner, va, rng)
                thr = best_f1_threshold(y[va], sv)
            else:
                thr = best_f1_threshold(y[tr], fit_score(name, X, y, tr, tr, rng))
            alarm = (s >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y[te], alarm, average="binary", zero_division=0)
            row = {"protocol": protocol, "horizon": horizon, "fold": f, "model": name,
                   "pr_auc": average_precision_score(y[te], s), "roc_auc": roc_auc_score(y[te], s),
                   "precision": p, "recall": r, "f1": f1, "prevalence": y[te].mean(), "fit_s": time.time() - t0}
            if protocol != "random":
                row.update(event_metrics(meta.iloc[te].reset_index(drop=True), y[te], alarm))
            rows.append(row)
    return pd.DataFrame(rows)
