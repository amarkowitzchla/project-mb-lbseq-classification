#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.base           import clone
from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing     import StandardScaler, label_binarize
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, f1_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    make_scorer, roc_curve, auc
)

# ───────────────────────── CONFIG
DATA_CSV     = "derived/medulloLPWGS_delfi5mb_tfx_CNV5mb_motifNMF7.2_fit_training.csv"
RANDOM_STATE = 42
N_FOLDS      = 3

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("derived", exist_ok=True)
os.makedirs("derived/binary_rocs_cv3", exist_ok=True)

# ───────────────────────── LOAD DATA
df_all = pd.read_csv(DATA_CSV).dropna(subset=["Sample_Class"])
X_full = df_all.drop("Sample_Class", axis=1)
y_full = df_all["Sample_Class"]

# ───────────────────────── TASKS
CLASSES = ["G3", "G4", "SHH", "WNT"]
tasks = {"G3G4_vs_Rest": lambda s: s.isin(["G3", "G4"]).astype(int)}
for c in CLASSES:
    tasks[f"{c}_vs_Rest"] = lambda s, c=c: (s == c).astype(int)
def pair_fn(s, a, b): return s.map({a: 1, b: 0})          # keep NaNs
for a, b in combinations(CLASSES, 2):
    tasks[f"{a}_vs_{b}"] = lambda s, a=a, b=b: pair_fn(s, a, b)

OVR   = [k for k in tasks if k.endswith("_vs_Rest")] + ["G3G4_vs_Rest"]
PAIR  = [k for k in tasks if "_vs_" in k and not k.endswith("_vs_Rest")]

# ───────────────────────── MODELS / GRIDS
estimators = {
    "LogisticRegression": LogisticRegression(
        penalty="l1", solver="saga", class_weight="balanced",
        max_iter=10_000, random_state=RANDOM_STATE),
    "SVM": LinearSVC(
        penalty="l2", dual=False, class_weight="balanced",
        max_iter=10_000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
param_grids = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10]},
    "SVM"              : {"clf__C": [0.01, 0.1, 1, 10]},
    "RandomForest"     : {"clf__n_estimators": [100, 200],
                          "clf__max_depth"   : [None, 10, 20]},
    "GradientBoosting" : {"clf__n_estimators": [100, 200],
                          "clf__learning_rate": [0.01, 0.1],
                          "clf__max_depth"   : [3, 5]},
}

cv_inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
scoring  = {"F1": make_scorer(f1_score, zero_division=0)}

# ───────────────────────── HELPERS
def safe_sampler(min_size):
    if min_size >= 6:
        k = max(1, min(3, min_size // 2))
        return SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    return RandomOverSampler(random_state=RANDOM_STATE)

def build_meta(X, y, pipes, chosen):
    meta = pd.DataFrame(index=X.index)
    for task, mname in chosen.items():
        pipe  = pipes[f"{task}__{mname}"]
        y_bin = tasks[task](y).dropna()         # ← drop here
        idx   = y_bin.index
        scores = (pipe.predict_proba(X.loc[idx])[:, 1]
                  if hasattr(pipe, "predict_proba")
                  else pipe.decision_function(X.loc[idx]))
        meta.loc[idx, f"{task}_score"] = scores
    return meta.fillna(0)

# ───────────────────────── OUTER CV
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
mean_fpr = np.linspace(0,1,401); fold_tprs, fold_aucs = [], []

# **NEW** collectors for binary ROC averaging
bin_mean_tprs = {t: [] for t in tasks}   # task → list[interpolated TPR]
bin_aucs      = {t: [] for t in tasks}

for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_full, y_full), 1):
    print(f"\n════════ Fold {fold}/{N_FOLDS} ════════")
    fold_dir = Path(f"derived/fold{fold}").mkdir(exist_ok=True, parents=True)

    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

    best_pipes, best_models = {}, {}

    # ─── train binary
    for task, labelfn in tasks.items():
        # ---- create binary labels and DROP NaNs right away
        y_raw = labelfn(y_tr)
        y_bin = y_raw.dropna()
        if y_bin.nunique() < 2:
            continue                                # need both classes

        X_bin = X_tr.loc[y_bin.index]
        sampler = safe_sampler(y_bin.value_counts().min())

        best_f1, best_name = -np.inf, None
        for mname, est in estimators.items():
            pipe = ImbPipeline([
                ("sampler", sampler),
                ("vth", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("clf", est)])
            grid = GridSearchCV(pipe, param_grids[mname],
                                cv=cv_inner, scoring="f1",
                                n_jobs=-1, verbose=0)
            grid.fit(X_bin, y_bin)
            best_pipes[f"{task}__{mname}"] = grid.best_estimator_
            if grid.best_score_ > best_f1:
                best_f1, best_name = grid.best_score_, mname
        if best_name:
            best_models[task] = best_name
            print(f"  {task:18s} → {best_name:17s} (F1={best_f1:.3f})")

    joblib.dump({"pipes":best_pipes,"best_models":best_models},
                f"derived/fold{fold}/binary_pipes.joblib")

    # ─── ROC for each binary on this fold’s validation split
    for task, mname in best_models.items():
        pipe   = best_pipes[f"{task}__{mname}"]
        y_bin  = tasks[task](y_va).dropna()
        if y_bin.nunique()<2: continue
        X_task = X_va.loc[y_bin.index]
        s      = (pipe.predict_proba(X_task)[:,1]
                  if hasattr(pipe,"predict_proba")
                  else pipe.decision_function(X_task))
        fpr, tpr, _ = roc_curve(y_bin, s)
        bin_mean_tprs[task].append(np.interp(mean_fpr, fpr, tpr))
        bin_aucs[task].append(auc(fpr, tpr))

    # ─── meta features / classifier (unchanged)
    X_meta_tr = build_meta(X_tr, y_tr, best_pipes, best_models)
    X_meta_va = build_meta(X_va, y_va, best_pipes, best_models)

    meta = LogisticRegression(multi_class="multinomial",
                              class_weight="balanced",
                              max_iter=10_000, random_state=RANDOM_STATE)
    grid = GridSearchCV(meta, {"C":[0.01,0.1,1,10]},
                        cv=cv_inner, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_meta_tr, y_tr)
    best_meta = clone(meta).set_params(**grid.best_params_).fit(X_meta_tr, y_tr)

    # ─── meta ROC fold
    prob = best_meta.predict_proba(X_meta_va)
    y_bin = label_binarize(y_va, classes=best_meta.classes_)
    mean_tpr = np.zeros_like(mean_fpr)
    cnt=0
    for i in range(y_bin.shape[1]):
        if len(np.unique(y_bin[:,i]))<2: continue
        fpr,tpr,_=roc_curve(y_bin[:,i],prob[:,i])
        mean_tpr+=np.interp(mean_fpr,fpr,tpr); cnt+=1
    mean_tpr/=cnt; auc_fold=auc(mean_fpr,mean_tpr)
    fold_tprs.append(mean_tpr); fold_aucs.append(auc_fold)

# ───────────────────────── AGGREGATE META ROC
fold_tprs = np.vstack(fold_tprs)
meta_mean = fold_tprs.mean(axis=0); meta_std = fold_tprs.std(axis=0)
meta_auc  = auc(mean_fpr, meta_mean)
plt.figure(figsize=(6,6))
plt.plot(mean_fpr, meta_mean, lw=2,
         label=f"Mean ROC (AUC={meta_auc:.2f})")
plt.fill_between(mean_fpr, np.maximum(meta_mean-meta_std,0),
                 np.minimum(meta_mean+meta_std,1), alpha=.25,label="±1 SD")
plt.plot([0,1],[0,1],'k--',lw=1); plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("3-Fold Averaged ROC – Meta-Classifier"); plt.legend()
plt.tight_layout(); plt.savefig("derived/meta_classifier_roc_cv3.png",dpi=300)

# ───────────────────────── COMPOSITE BINARY ROC PLOTS
def composite(task_list, title, fname):
    cmap = plt.cm.get_cmap("tab10", len(task_list))
    plt.figure(figsize=(6,6))
    for i, t in enumerate(task_list):
        if not bin_mean_tprs[t]: continue
        mean_tpr = np.mean(bin_mean_tprs[t], axis=0)
        mean_auc = np.mean(bin_aucs[t])
        plt.plot(mean_fpr, mean_tpr, color=cmap(i), lw=2,
                 label=f"{t} (AUC={mean_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(fontsize="small")
    plt.tight_layout()
    out = f"derived/binary_rocs_cv3/{fname}"
    plt.savefig(out,dpi=300); plt.close()
    print(f"Saved composite ROC → {out}")

composite(OVR,   "OvR Tasks – 5-Fold Avg",  "summary_ovr.png")
composite(PAIR,  "Pairwise Tasks – 5-Fold Avg", "summary_pairwise.png")

print(f"\nMeta AUC (mean of folds) = {meta_auc:.3f}")
