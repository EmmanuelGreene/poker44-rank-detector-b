"""Load cached Poker44 benchmark releases into labeled chunk-group examples.

LOCAL ONLY (gitignored). Reads the SHARED per-fleet cache (one download per
VPS, all miners train from it). Resolution order: $POKER44_TRAIN_DATA_DIR,
then <fleet folder>/train_data next to this repo, then ./data_cache.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def default_data_dir() -> str:
    env = os.environ.get("POKER44_TRAIN_DATA_DIR", "").strip()
    if env:
        return env
    shared = os.path.join(os.path.dirname(_REPO), "train_data")
    if os.path.isdir(shared):
        return shared
    return os.path.join(_HERE, "data_cache")


@dataclass
class Example:
    hands: List[Dict[str, Any]]   # the chunk group (miner-visible hands)
    label: int                    # 1 = bot, 0 = human
    source_date: str
    split: str
    chunk_id: str
    group_index: int


def load_examples(data_dir: str | None = None) -> List[Example]:
    data_dir = data_dir or default_data_dir()
    examples: List[Example] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        try:
            data = doc["data"]
            source_date = data["sourceDate"]
            records = data["chunks"]
        except Exception:
            continue          # malformed cache file; skip, never abort the load
        for record in records:
            try:
                split = record.get("split", "train")
                chunk_id = record.get("chunkId", "")
                groups = record["chunks"]
                labels = record["groundTruth"]
            except Exception:
                continue
            for gi, (group, label) in enumerate(zip(groups, labels)):
                examples.append(Example(hands=group, label=int(label),
                                        source_date=source_date, split=split,
                                        chunk_id=chunk_id, group_index=gi))
    return examples


if __name__ == "__main__":
    from collections import Counter
    ex = load_examples()
    print(f"data_dir: {default_data_dir()}")
    print(f"total examples: {len(ex)}")
    print("label balance:", Counter(e.label for e in ex))
    print("dates:", len(set(e.source_date for e in ex)))
