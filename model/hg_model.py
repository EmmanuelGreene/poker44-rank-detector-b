"""poker44-rank-detector-b — HG2Blend: weighted-rank blended bot detector.

Family: high-capacity gradient trio. Three decorrelated members vote by rank
with fixed weights chosen on walk-forward validation:
  * a stacked gradient ensemble on the tree feature view;
  * a monotone-constrained booster restricted to sign-stable features;
  * a neural PCA->MLP on the wide (union) feature view.
Rank voting is calibration-free and robust to member score-scale drift.
Blend weights: (0.35, 0.30, 0.35).
"""
import numpy as np


class HG2Blend:
    def __init__(self, stack, mono, mlp, cols_tree, cols_wide, weights=(0.35, 0.30, 0.35)):
        self.stack = stack
        self.mono = mono
        self.mlp = mlp
        self.cols_tree = list(cols_tree)
        self.cols_wide = list(cols_wide)
        self.weights = tuple(float(w) for w in weights)

    @staticmethod
    def _rank01(s):
        s = np.asarray(s, dtype=float)
        if s.size <= 1:
            return np.zeros_like(s)
        return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)

    def score(self, Xtree, Xwide):
        ws, wm, wp = self.weights
        ss = self.stack.predict_proba(Xtree)[:, 1]
        ms = self.mono.predict_proba(Xtree)[:, 1]
        ps = self.mlp.predict_proba(Xwide)[:, 1]
        num = ws * self._rank01(ss) + wm * self._rank01(ms) + wp * self._rank01(ps)
        return num / (ws + wm + wp)
