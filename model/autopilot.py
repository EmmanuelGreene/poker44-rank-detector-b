#!/usr/bin/env python3
"""Guarded auto-retrain for poker44-rank-detector-b. LOCAL ONLY — gitignored.

One run = REFRESH (pull any new daily benchmark release into the SHARED fleet
cache) -> RETRAIN (train_hg.py) -> GUARD (accept only if walk-forward
reward does not regress and FPR stays under the deploy ceiling; else revert
the artifact backup) -> DEPLOY (pm2 restart so the miner serves the new
artifact). Idempotent and crash-safe: the previous artifact is snapshotted
before training and restored on any failure or regression.

Normally invoked by the fleet-level fleet_retrain.py with --no-refresh.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


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

ART = os.path.join(HERE, "artifacts")
STAGING = os.path.join(HERE, "artifacts_staging")
BACKUPS = os.path.join(HERE, "artifacts_backups")
LOG = os.path.join(HERE, "autopilot.log")
HISTORY = os.path.join(HERE, "autopilot_history.jsonl")
DATA = (os.environ.get("POKER44_TRAIN_DATA_DIR", "").strip()
        or os.path.join(os.path.dirname(REPO), "train_data"))
ARTIFACT = os.environ.get("POKER44_ARTIFACT", "rank_detector_b_v32.pkl")
TRAIN_SCRIPT = os.path.join(HERE, "train_hg.py")
API = os.environ.get("POKER44_BENCHMARK_REPO_URL",
                     "https://api.poker44.net/api/v1/benchmark").rstrip("/")
PY = sys.executable

REWARD_EPSILON = 0.002
MAX_DEPLOY_FPR = float(os.environ.get("POKER44_MAX_DEPLOY_FPR", "0.055"))
MINER_PM2_NAME = os.environ.get("POKER44_PM2_NAME", "poker44_high_2")


def log(msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def _get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "poker44-hg-autopilot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def refresh_data():
    os.makedirs(DATA, exist_ok=True)
    try:
        doc = json.loads(_get(f"{API}/releases?limit=100"))
        releases = [r["sourceDate"] for r in doc["data"]["releases"]]
    except Exception as exc:
        log(f"REFRESH: could not list releases ({exc}); using cached data only")
        return 0
    added = 0
    for d in sorted(releases):
        path = os.path.join(DATA, f"{d}.json")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        try:
            blob = _get(f"{API}/chunks?sourceDate={d}&limit=48")
            parsed = json.loads(blob)
            if "data" not in parsed or "chunks" not in parsed["data"]:
                log(f"REFRESH: {d} payload missing data.chunks; skipping")
                continue
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
            added += 1
            log(f"REFRESH: cached new date {d}")
        except Exception as exc:
            log(f"REFRESH: failed to fetch {d} ({exc})")
    log(f"REFRESH: {added} new date(s) in shared cache {DATA}")
    return added


def read_meta(art_dir=ART):
    try:
        with open(os.path.join(art_dir, "meta.json")) as fh:
            return json.load(fh)
    except Exception:
        return None


def backup_current():
    if not os.path.exists(os.path.join(ART, ARTIFACT)):
        return None
    os.makedirs(BACKUPS, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(BACKUPS, stamp)
    shutil.copytree(ART, dest)
    snaps = sorted(d for d in os.listdir(BACKUPS)
                   if os.path.isdir(os.path.join(BACKUPS, d)))
    for old in snaps[:-10]:
        shutil.rmtree(os.path.join(BACKUPS, old), ignore_errors=True)
    return dest


def record_history(decision, old_reward, new_reward, fpr, n_dates, reasons):
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "decision": decision,
           "old_reward": old_reward, "new_reward": new_reward, "fpr": fpr,
           "n_dates": n_dates, "reasons": reasons}
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def retrain_and_guard(force_deploy):
    """Train a CANDIDATE into a staging dir; only an accepted candidate is
    atomically swapped into the serving artifacts dir. The live artifact is
    never touched during training, so a mid-train miner (re)start always
    loads a validated model."""
    old_meta = read_meta(ART)
    old_reward = float(old_meta["cv_reward"]) if old_meta else -1.0
    old_dates = int(old_meta.get("n_dates", 0)) if old_meta else 0
    shutil.rmtree(STAGING, ignore_errors=True)
    os.makedirs(STAGING, exist_ok=True)
    log(f"RETRAIN: baseline cv_reward={old_reward:.4f} over {old_dates} dates; "
        "training candidate into staging")

    proc = subprocess.run([PY, TRAIN_SCRIPT], cwd=HERE, capture_output=True,
                          text=True,
                          env={**os.environ, "POKER44_REPO": REPO,
                               "POKER44_ART_DIR": STAGING,
                               "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        log(f"RETRAIN: trainer FAILED rc={proc.returncode} (serving artifact untouched)")
        log(proc.stderr.strip()[-1500:])
        return False

    new_meta = read_meta(STAGING)
    if not new_meta or not os.path.exists(os.path.join(STAGING, ARTIFACT)):
        log("RETRAIN: staging incomplete after training (serving artifact untouched)")
        return False

    new_reward = float(new_meta["cv_reward"])
    new_fpr = float(new_meta.get("cv_fpr", 1.0))
    new_dates = int(new_meta.get("n_dates", 0))
    log(f"RETRAIN: candidate cv_reward={new_reward:.4f} cv_fpr={new_fpr:.4f} "
        f"cv_ap={new_meta.get('cv_ap', 0):.4f} over {new_dates} dates")

    reasons = []
    if new_fpr >= MAX_DEPLOY_FPR:
        reasons.append(f"fpr {new_fpr:.4f} >= ceiling {MAX_DEPLOY_FPR}")
    if not force_deploy and new_reward < old_reward - REWARD_EPSILON:
        reasons.append(f"reward {new_reward:.4f} < baseline {old_reward:.4f} - eps")

    if reasons and old_meta:
        log("RETRAIN: REJECTED (" + "; ".join(reasons) + ") -> candidate discarded")
        record_history("rejected", old_reward, new_reward, new_fpr, new_dates, reasons)
        return False
    if reasons:
        log("RETRAIN: WARNING first model violates a guard but nothing is serving "
            "yet, deploying anyway: " + "; ".join(reasons))

    backup = backup_current()
    os.makedirs(ART, exist_ok=True)
    os.replace(os.path.join(STAGING, ARTIFACT), os.path.join(ART, ARTIFACT))
    os.replace(os.path.join(STAGING, "meta.json"), os.path.join(ART, "meta.json"))
    log(f"RETRAIN: PROMOTED cv_reward {old_reward:.4f} -> {new_reward:.4f}"
        f"{' (previous artifact backed up)' if backup else ''}")
    record_history("promoted", old_reward, new_reward, new_fpr, new_dates, [])
    return True


def restart_miner():
    try:
        out = subprocess.run(["pm2", "restart", MINER_PM2_NAME, "--update-env"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            log(f"DEPLOY: restarted pm2 process '{MINER_PM2_NAME}'")
        else:
            log(f"DEPLOY: pm2 restart failed rc={out.returncode}: "
                f"{out.stderr.strip()[-400:]}")
    except FileNotFoundError:
        log("DEPLOY: pm2 not found on PATH; restart the miner manually")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--force-deploy", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log("=== AUTOPILOT START (poker44-rank-detector-b) ===")
    added = 0 if args.no_refresh else refresh_data()
    promoted = retrain_and_guard(force_deploy=args.force_deploy)
    if promoted and not args.no_restart:
        restart_miner()
    elif promoted:
        log("DEPLOY: promoted but --no-restart set; serving stale until restart")
    else:
        log("DEPLOY: nothing to deploy (model unchanged)")
    log(f"=== AUTOPILOT DONE in {time.time() - t0:.0f}s | new_dates={added} "
        f"promoted={promoted} ===")


if __name__ == "__main__":
    main()
