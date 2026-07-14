#!/usr/bin/env python3
"""Training pipeline for poker44-rank-detector-b (HG2Blend). LOCAL ONLY — gitignored.

Data: the shared daily public-benchmark cache. Validation: honest walk-forward
(train strictly-past dates -> test the next unseen date, REAL rows only).
Live-robustness: pooled/subset augmentation teaches validator-size groups
(live groups run ~80-105 hands vs 30-40 in the benchmark). Deployment
threshold is set on the human-score quantile at the target FPR. The artifact
is written atomically so a serving miner never sees a half-written model.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_env(path):
    try:
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.split(" #", 1)[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in (chr(34), chr(39)):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    except FileNotFoundError:
        pass


_load_env(os.path.join(REPO, ".env"))

from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from dataset import load_examples  # noqa: E402
from evaluate import fpr_target_threshold  # noqa: E402
from reward_fn import reward  # noqa: E402
from hg_features import tree_view, wide_view  # noqa: E402
from hg_model import HG2Blend  # noqa: E402

ART = os.environ.get("POKER44_ART_DIR", "").strip() or os.path.join(HERE, "artifacts")
ARTIFACT = os.environ.get("POKER44_ARTIFACT", "rank_detector_b_v32.pkl")
TARGET_FPR = float(os.environ.get("POKER44_TARGET_FPR", "0.035"))
NJ = int(os.environ.get("POKER44_TRAIN_JOBS", "4"))
WF = int(os.environ.get("POKER44_WF_POINTS", "3"))
SEED = 1200

# live-size augmentation knobs (per-miner)
POOL_RANGE = (90, 105)          # pooled-group target size (min, max) hands
POOL_PER_DATE = 3           # pooled groups per (date, label)
SUBSET_RANGE = None      # contiguous subset size range, or None
SUBSET_PER_DATE = 0


def sanitize(hands):
    out = []
    for h in hands:
        try:
            out.append(prepare_hand_for_miner(h))
        except Exception:
            out.append(h)
    return out


def augment(san, y, dates):
    """Pooled (live-size) and subset resamples of PUBLIC benchmark groups only."""
    rng = random.Random(SEED)
    aug_chunks, aug_y, aug_dates = [], [], []
    by_key = {}
    for i, (d, lab) in enumerate(zip(dates, y)):
        by_key.setdefault((d, int(lab)), []).append(i)
    for (d, lab), idxs in sorted(by_key.items()):
        if len(idxs) < 2:
            continue
        for _ in range(POOL_PER_DATE):
            target = rng.randint(*POOL_RANGE)
            pool, used = [], 0
            for i in rng.sample(idxs, len(idxs)):
                pool.extend(san[i])
                used += 1
                if len(pool) >= target:
                    break
            if used >= 2 and len(pool) >= POOL_RANGE[0]:
                aug_chunks.append(pool[:target])
                aug_y.append(lab)
                aug_dates.append(d)
        if SUBSET_RANGE:
            for _ in range(SUBSET_PER_DATE):
                i = rng.choice(idxs)
                take = min(rng.randint(*SUBSET_RANGE), len(san[i]))
                if take >= 8 and len(san[i]) > take:
                    start = rng.randint(0, len(san[i]) - take)
                    aug_chunks.append(san[i][start:start + take])
                    aug_y.append(lab)
                    aug_dates.append(d)
    return aug_chunks, np.asarray(aug_y, dtype=int), np.asarray(aug_dates)


def mat(chunks, view_fn, cols=None):
    X = pd.DataFrame([view_fn(c) for c in chunks]).fillna(0.0)
    if cols is None:
        cols = sorted(X.columns)
    return X.reindex(columns=cols, fill_value=0.0).values.astype(float), list(cols)


def mono_vector(X, y, dates):
    """Per-feature monotone sign, kept only when sign-stable across dates."""
    ud = sorted(set(dates.tolist()))
    out = []
    for j in range(X.shape[1]):
        sg = []
        for d in ud:
            m = dates == d
            if m.sum() >= 8 and len(set(y[m].tolist())) > 1:
                r = spearmanr(X[m, j], y[m]).correlation
                if r is not None and not np.isnan(r):
                    sg.append(r)
        ok = (len(sg) >= 5 and abs(float(np.mean(sg))) >= 0.05
              and float((np.sign(sg) == np.sign(np.mean(sg))).mean()) >= 0.7)
        out.append(int(np.sign(np.mean(sg))) if ok else 0)
    return out


def rk(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def atomic_write(payload_bytes, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload_bytes)
    os.replace(tmp, path)


def build_stack():
    base = [("lgb", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02,
                                       num_leaves=127, n_jobs=NJ,
                                       random_state=SEED, verbose=-1)),
            ("xgl", xgb.XGBClassifier(n_estimators=600, learning_rate=0.03,
                                      grow_policy="lossguide", max_leaves=64, max_depth=0,
                                      tree_method="hist", n_jobs=NJ,
                                      random_state=SEED + 1, eval_metric="logloss")),
            ("cat", cb.CatBoostClassifier(iterations=800, learning_rate=0.025, depth=7,
                                          verbose=0, thread_count=NJ,
                                          random_seed=SEED + 2, allow_writing_files=False)),
            ("rf", RandomForestClassifier(n_estimators=600, max_depth=18, n_jobs=NJ,
                                          random_state=SEED + 3,
                                          class_weight="balanced_subsample"))]
    return StackingClassifier(base, final_estimator=LogisticRegression(C=0.5, max_iter=1000),
                              cv=4, n_jobs=1)


def build_mono(mono):
    return VotingClassifier([(f"l{i}", lgb.LGBMClassifier(
        n_estimators=550, learning_rate=0.03, num_leaves=31,
        monotone_constraints=list(int(c) for c in mono),
        n_jobs=NJ, random_state=SEED + 10 + i, verbose=-1)) for i in range(3)],
        voting="soft", n_jobs=1)


def build_mlp():
    return VotingClassifier([(f"m{i}", Pipeline([
        ("s", StandardScaler()), ("p", PCA(56, random_state=SEED)),
        ("m", MLPClassifier((80,), alpha=2.0, max_iter=700, early_stopping=True,
                            validation_fraction=0.15, n_iter_no_change=15,
                            random_state=SEED + 20 + i))])) for i in range(3)],
        voting="soft", n_jobs=1)

BLEND_W = (0.35, 0.30, 0.35)


def blend(parts):
    return sum(w * rk(p) for w, p in zip(BLEND_W, parts)) / sum(BLEND_W)


def fit_members(Xtree, Xwide, yv, mask, mono):
    stack = build_stack().fit(Xtree[mask], yv[mask])
    monom = build_mono(mono).fit(Xtree[mask], yv[mask])
    mlpm = build_mlp().fit(Xwide[mask], yv[mask])
    return stack, monom, mlpm


def predict_members(models, Xtree, Xwide, mask):
    stack, monom, mlpm = models
    return [stack.predict_proba(Xtree[mask])[:, 1],
            monom.predict_proba(Xtree[mask])[:, 1],
            mlpm.predict_proba(Xwide[mask])[:, 1]]

if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    ex = load_examples()
    if not ex:
        print("no training data found - check POKER44_TRAIN_DATA_DIR", file=sys.stderr)
        sys.exit(2)
    san = [sanitize(e.hands) for e in ex]
    y = np.asarray([e.label for e in ex], dtype=int)
    dates = np.asarray([e.source_date for e in ex])
    aug_chunks, aug_y, aug_dates = augment(san, y, dates)
    all_chunks = san + aug_chunks
    ally = np.concatenate([y, aug_y]) if len(aug_y) else y
    alldates = np.concatenate([dates, aug_dates]) if len(aug_dates) else dates
    is_real = np.zeros(len(all_chunks), dtype=bool)
    is_real[:len(san)] = True

    TREE, cols_tree = mat(all_chunks, tree_view)
    WIDE, cols_wide = mat(all_chunks, wide_view)
    mono = mono_vector(TREE[is_real], y, dates)
    ud = sorted(set(dates.tolist()))
    print(f"poker44-rank-detector-b: {len(y)} real + {len(aug_y)} aug chunks | "
          f"tree{TREE.shape[1]} wide{WIDE.shape[1]} | {sum(1 for c in mono if c)} monotone | {len(ud)} dates ({time.time() - t0:.0f}s)", flush=True)

    oof = np.full(len(y), np.nan)
    for td in ud[-WF:]:
        tr = alldates < td
        te_real = dates == td
        te = np.zeros(len(all_chunks), dtype=bool)
        te[:len(san)] = te_real
        if tr.sum() < 60 or len(set(ally[tr].tolist())) < 2 or not te.any():
            continue
        tr_real = tr & is_real
        mono_tr = mono_vector(TREE[tr_real], ally[tr_real], alldates[tr_real])
        models = fit_members(TREE, WIDE, ally, tr, mono_tr)
        oof[te_real] = blend(predict_members(models, TREE, WIDE, te))
        print(f"  wf {td} ({time.time() - t0:.0f}s)", flush=True)

    m = ~np.isnan(oof)
    if not m.any():
        print("walk-forward produced no scores; refusing to deploy blind", file=sys.stderr)
        sys.exit(3)
    cv_ap = float(average_precision_score(y[m], oof[m]))
    rew, res = reward(oof[m], y[m])
    deploy_t = fpr_target_threshold(oof[m][y[m] == 0], TARGET_FPR)
    print(f"WALK-FORWARD[{WF}d]: cv_ap={cv_ap:.4f} reward={rew:.4f} "
          f"recall@fpr={res['bot_recall']:.3f} fpr={res['fpr']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    stack, monom, mlpm = fit_members(TREE, WIDE, ally,
                                     np.ones(len(all_chunks), dtype=bool), mono)
    ens = HG2Blend(stack, monom, mlpm, cols_tree, cols_wide, weights=BLEND_W)

    meta = {
        "model_name": "poker44-rank-detector-b",
        "model_class": "HG2Blend",
        "family": "hg",
        "model": "Weighted-rank trio with a leaf-wise stack (LightGBM-127/lossguide-XGBoost/deep CatBoost-d7/RandomForest-d18 -> logistic meta), a 3-seed monotone-constrained LightGBM, and a 3-seed PCA(56)->MLP(80) member; tightest FPR target of the family (0.035).",
        "feature_version": "hg.v1",
        "trained_on": "sanitized (prepare_hand_for_miner; train == serve)",
        "deploy_threshold": float(deploy_t),
        "target_fpr": TARGET_FPR,
        "seed": SEED,
        "cv_ap": cv_ap,
        "cv_reward": float(rew),
        "cv_recall": float(res["bot_recall"]),
        "cv_fpr": float(res["fpr"]),
        "validation": f"walk-forward over the last {WF} dates (train past -> test next unseen date)",
        "reward_formula": "0.75*AP + 0.25*recall@fpr<=0.05 (official 2026-06-26)",
        "n_train_real": int(len(y)),
        "n_train_aug": int(len(aug_y)),
        "augmentation": {"pool_range": list(POOL_RANGE), "pool_per_date": POOL_PER_DATE,
                          "subset_range": list(SUBSET_RANGE) if SUBSET_RANGE else None,
                          "subset_per_date": SUBSET_PER_DATE},
        "blend_weights": list(BLEND_W),
        "n_features_tree": int(TREE.shape[1]),
        "n_features_wide": int(WIDE.shape[1]),
        "n_monotone": int(sum(1 for c in mono if c)),
        "n_dates": int(len(ud)),
        "benchmark_releases": sorted(set(dates.tolist())),
        "artifact": ARTIFACT,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(pickle.dumps(ens), os.path.join(ART, ARTIFACT))
    atomic_write(json.dumps(meta, indent=2).encode("utf-8"),
                 os.path.join(ART, "meta.json"))
    print(f"saved {ARTIFACT} + meta.json | cv_ap={cv_ap:.4f} cv_reward={rew:.4f}", flush=True)
