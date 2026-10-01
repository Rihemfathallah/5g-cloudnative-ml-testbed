"""Run folders, frozen manifests and seeding.

runs/<run_id>/
  manifest.yaml     the run config exactly as executed + git commit + library versions
  labels.csv        ground truth
  context.jsonl     traffic events
  raw.parquet       canonical time series
  windows.parquet   features + labels + split
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .config import REPO_ROOT


def make_run_id(name: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return f"{now:%Y-%m-%d_%H%M%S}_{name}"


def run_dir(run_id: str, root: str | Path | None = None) -> Path:
    root = Path(root) if root else REPO_ROOT / "runs"
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_state() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "torch", "scipy"):
        try:
            out[mod] = str(__import__(mod).__version__)
        except Exception:
            out[mod] = None
    return out


def freeze_manifest(cfg: dict, out_dir: Path, run_id: str, extra: dict | None = None) -> Path:
    """Write the config as executed. A dirty working tree is recorded, not refused, so that
    exploratory runs are possible -- but the report should only cite runs with dirty: false."""
    doc = {
        "run_id": run_id,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_state(),
        "versions": library_versions(),
        "host": platform.node(),
        "config": cfg,
    }
    if extra:
        doc.update(extra)
    p = out_dir / "manifest.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def derive_seed(base: int, *names: str) -> int:
    """Independent, reproducible sub-seeds: traffic and faults must not share a stream, or adding
    one traffic episode type would shift every fault time."""
    h = np.random.SeedSequence([base, *[int.from_bytes(n.encode(), "little") % (2**32) for n in names]])
    return int(h.generate_state(1)[0])


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
