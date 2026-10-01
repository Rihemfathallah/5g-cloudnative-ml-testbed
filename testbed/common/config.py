"""Config loading and small parsing helpers shared by every component."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\s*$")
_UNIT_S = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_duration(value: Any) -> float:
    """Parse '90s', '15m', '1.5h', '200ms' or a bare number (seconds) into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _DURATION_RE.match(str(value))
    if not m:
        raise ValueError(f"cannot parse duration {value!r} (use e.g. 30s, 15m, 2h)")
    return float(m.group(1)) * _UNIT_S[m.group(2)]


def parse_range(value: Any) -> tuple[float, float]:
    """A duration or [lo, hi] duration range -> (lo_s, hi_s)."""
    if isinstance(value, (list, tuple)):
        lo, hi = (parse_duration(v) for v in value)
    else:
        lo = hi = parse_duration(value)
    if hi < lo:
        raise ValueError(f"range {value!r} has hi < lo")
    return lo, hi


def parse_size_bytes(value: Any) -> int:
    m = re.match(r"^\s*(\d+)\s*(B|KB|MB|GB)\s*$", str(value), re.I)
    if not m:
        raise ValueError(f"cannot parse size {value!r}")
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[m.group(2).upper()]
    return int(m.group(1)) * mult


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
