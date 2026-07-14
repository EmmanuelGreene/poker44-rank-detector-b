"""poker44-rank-detector-b — Poker44 (SN126) bot-detection miner entrypoint.

Serves the HG2Blend detector and attaches a model manifest (repo, commit,
implementation-file hashes, artifact digest, data attestations) to every
response so validators can verify the model identity end-to-end.

All deployment identity (wallet, hotkey, port, repo url/commit, artifact and
threshold knobs) comes from the repo-local .env file; nothing is hardcoded.
"""

# NOTE: do NOT `from __future__ import annotations` here. bittensor's
# axon.attach introspects the real type of forward()'s `synapse` parameter via
# issubclass(); stringized annotations break that.

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("POKER44_REPO", "").strip() or os.path.dirname(MODEL_DIR)
for _p in (REPO_DIR, MODEL_DIR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def _load_env(path: str) -> None:
    """Fill os.environ from .env without overriding already-exported values."""
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


_load_env(os.path.join(REPO_DIR, ".env"))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (build_local_model_manifest,
                                          evaluate_manifest_compliance,
                                          manifest_digest)
from poker44.validator.synapse import DetectionSynapse

from infer import artifact_sha256, get_model

# optional local synapse capture (live-traffic monitoring); disabled unless the
# private .env points at a sink implementation; never blocks serving
_SINK_DIR = os.environ.get("POKER44_SYNAPSE_SINK_DIR", "").strip()
try:
    if not _SINK_DIR:
        raise ImportError("synapse sink not configured")
    if _SINK_DIR not in sys.path:
        sys.path.insert(0, _SINK_DIR)
    from synapse_sink import record_synapse
except Exception:  # capture is optional
    def record_synapse(*_a, **_k):
        return None


def _repo_commit() -> str:
    commit = os.environ.get("POKER44_MODEL_REPO_COMMIT", "").strip()
    if commit:
        return commit
    try:
        proc = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip()
    except Exception:
        return ""


class MLMiner(BaseMinerNeuron):
    """poker44-rank-detector-b: lossguide-stack4 + monotone-lgbm + pca56-mlp80 (weighted-rank 35/30/35)."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.poker_model = get_model()
        meta = self.poker_model.meta
        repo_url = os.environ.get("POKER44_MODEL_REPO_URL", "").strip().rstrip("/")
        model_card = f"{repo_url}#readme" if repo_url.count("/") >= 4 else ""
        self.model_manifest = build_local_model_manifest(
            repo_root=Path(REPO_DIR),
            implementation_files=[
                Path(MODEL_DIR) / "poker44_miner.py",
                Path(MODEL_DIR) / "infer.py",
                Path(MODEL_DIR) / "hg_features.py",
                Path(MODEL_DIR) / "hg_model.py",
                Path(MODEL_DIR) / "features_v2.py",
                Path(MODEL_DIR) / "poker44_ml" / "__init__.py",
                Path(MODEL_DIR) / "poker44_ml" / "features.py",
            ],
            defaults={
                "model_name": os.environ.get("POKER44_MODEL_NAME", "poker44-rank-detector-b"),
                "model_version": os.environ.get("POKER44_MODEL_VERSION", "2.2-b"),
                "framework": "lossguide-stack4 + monotone-lgbm + pca56-mlp80 (weighted-rank 35/30/35)",
                "license": "MIT",
                "repo_url": repo_url,
                "repo_commit": _repo_commit(),
                "artifact_sha256": artifact_sha256(),
                "artifact_filename": self.poker_model.artifact_name,
                "model_card_url": model_card,
                "notes": ("Weighted-rank trio with a leaf-wise stack (LightGBM-127/lossguide-XGBoost/deep CatBoost-d7/RandomForest-d18 -> logistic meta), a 3-seed monotone-constrained LightGBM, and a 3-seed PCA(56)->MLP(80) member; tightest FPR target of the family (0.035). "
                          f"Walk-forward: ap={meta.get('cv_ap', 0.0):.4f} "
                          f"reward={meta.get('cv_reward', 0.0):.4f} "
                          f"over {meta.get('n_dates', 0)} benchmark dates. "
                          "Weights withheld from the repo; artifact identity is "
                          "pinned by artifact_sha256 in this manifest."),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the PUBLIC Poker44 benchmark releases "
                    "(api.poker44.net/api/v1/benchmark) plus size-resampled "
                    "augmentations of those same public groups. "
                    "No validator-only data is used."),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This model does not train on validator-only evaluation data."),
                "data_attestation": (
                    "Features use only miner-visible behavioral fields; no hole "
                    "cards, board cards, outcomes, or player identifiers."),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"poker44-rank-detector-b ready | cv_ap={meta.get('cv_ap', 0.0):.4f} "
            f"cv_reward={meta.get('cv_reward', 0.0):.4f} "
            f"threshold={self.poker_model.threshold:.4f}")
        bt.logging.info(
            f"Manifest transparency: {self.manifest_compliance['status']} "
            f"(missing={self.manifest_compliance['missing_fields']}, "
            f"violations={self.manifest_compliance['policy_violations']}) "
            f"digest={manifest_digest(self.model_manifest)}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        try:
            scores = self.poker_model.score_chunks(chunks)
        except Exception as exc:  # never crash on a malformed request
            bt.logging.warning(f"scoring failed ({exc}); benign fallback 0.1")
            scores = [0.1] * len(chunks)  # never accuse anyone on an internal error
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | bots={sum(synapse.predictions)} "
            f"mean={sum(scores) / max(len(scores), 1):.3f}")
        record_synapse(synapse, miner="poker44-rank-detector-b")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with MLMiner() as miner:
        bt.logging.info("poker44-rank-detector-b miner running...")
        while True:
            try:
                bt.logging.info(
                    f"UID {miner.uid} | incentive {miner.metagraph.I[miner.uid]:.6f}")
            except Exception:
                pass
            time.sleep(5 * 60)
