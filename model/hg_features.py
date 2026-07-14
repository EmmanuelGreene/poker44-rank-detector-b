"""Feature views for poker44-rank-detector-b (published; train == serve exactly).

tree_view: per-chunk behavioral aggregates + bucket/entropy fingerprints from
           the base library, plus chunk-size descriptors (raw and log) so the
           model learns the group-size axis (live groups are larger than
           benchmark groups).
wide_view: tree_view merged with the v2 order-statistic aggregates into one
           deduplicated dictionary (the neural member consumes this union).
"""
import math

from features_v2 import extract_features_v2
from poker44_ml.features import chunk_features


def tree_view(chunk):
    hands = chunk or []
    d = chunk_features(hands)
    n = float(len(hands))
    d["hand_count"] = n
    d["hand_count_log"] = math.log1p(n)
    return d


def wide_view(chunk):
    d = dict(extract_features_v2(chunk or []))
    d.update(tree_view(chunk))
    return d
