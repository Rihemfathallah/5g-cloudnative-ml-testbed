"""Mobility and signal traces.

A trace set (one speed, one split) is stored as one .npz:
  t_s      (T,)            sample times
  pos      (N, T, 2)       UE positions
  rsrp     (N, T, C)       noise-free large-scale RSRP, dBm/RE  -> SINR, labels
  meas     (N, T, C)       rsrp + L1 measurement error         -> what the UE reports (before L3)
  speed    (N,)            km/h
Splits are generated from DIFFERENT seeds: a trajectory never appears in two splits.

Two producers write this format: the pure-Python TR 38.901 model below, and the ns-3 program
in ns3/nr-rsrp-traces.cc via `from_ns3_csv`. The CV must name the producer actually used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from testbed.common.runinfo import derive_seed
from testbed.handover.channel import hex_sites, large_scale_rsrp

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def trajectory(rng, speed_kmh, duration_s, dt_s, radius_m, turn_every_s, turn_sigma_deg):
    """Straight segments with random heading changes; reflected at a circular boundary."""
    T = int(round(duration_s / dt_s))
    v = speed_kmh / 3.6
    r0 = radius_m * np.sqrt(rng.uniform(0, 0.8))
    a0 = rng.uniform(0, 2 * np.pi)
    p = np.array([r0 * np.cos(a0), r0 * np.sin(a0)])
    head = rng.uniform(0, 2 * np.pi)
    out = np.empty((T, 2))
    turn_every = max(1, int(round(turn_every_s / dt_s)))
    for k in range(T):
        out[k] = p
        if k and k % turn_every == 0:
            head += np.deg2rad(rng.normal(0, turn_sigma_deg))
        nxt = p + v * dt_s * np.array([np.cos(head), np.sin(head)])
        if np.linalg.norm(nxt) > radius_m:          # specular reflection on the circle
            n = nxt / np.linalg.norm(nxt)
            d = np.array([np.cos(head), np.sin(head)])
            d = d - 2 * np.dot(d, n) * n
            head = np.arctan2(d[1], d[0])
            nxt = p + v * dt_s * d
        p = nxt
    return out


def generate_set(cfg: dict, speed: float, split: str, n: int, base_seed: int) -> dict:
    dep, ch, mob = cfg["deployment"], cfg["channel"], cfg["mobility"]
    rng = np.random.default_rng(derive_seed(base_seed, "traces", split, str(speed)))
    dt = mob["sample_ms"] / 1000
    sites = hex_sites(dep["n_cells"], dep["isd_m"])
    pos, rsrp, meas = [], [], []
    for _ in range(n):
        p = trajectory(rng, speed, mob["duration_s"], dt, dep["area_radius_m"],
                       mob["turn_every_s"], mob["turn_sigma_deg"])
        r, _ = large_scale_rsrp(p, sites, dep, ch, rng)
        pos.append(p)
        rsrp.append(r)
        meas.append(r + rng.normal(0, ch.get("meas_noise_db", 1.0), r.shape))
    T = len(pos[0])
    return {"t_s": np.arange(T) * dt, "pos": np.array(pos, np.float32),
            "rsrp": np.array(rsrp, np.float32), "meas": np.array(meas, np.float32),
            "speed": np.full(n, speed, np.float32), "sites": sites}


def build_all(cfg: dict, out: Path, base_seed: int = 0, n_override: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for speed in cfg["mobility"]["speeds_kmh"]:
        for split in SPLITS:
            n = (n_override or cfg["mobility"]["n_traj"])[split]
            p = out / f"{split}_{int(speed)}kmh.npz"
            if not p.exists():
                np.savez_compressed(p, **generate_set(cfg, float(speed), split, n, base_seed))
                log.info("traces %s (%d traj)", p.name, n)
            paths[(split, speed)] = p
    return paths


def load(p: Path) -> dict:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def from_ns3_csv(csv_path: Path, meas_noise_db: float, seed: int) -> dict:
    """Convert ns-3 output (columns: traj, time_s, speed_kmh, x, y, cell, rsrp_dbm) to a trace set.
    The L1 measurement error is added here, identically to the Python producer."""
    df = pd.read_csv(csv_path)
    trajs = sorted(df["traj"].unique())
    cells = sorted(df["cell"].unique())
    times = np.sort(df["time_s"].unique())
    piv = df.pivot_table(index=["traj", "time_s"], columns="cell", values="rsrp_dbm")
    piv = piv.reindex(pd.MultiIndex.from_product([trajs, times]))
    if piv.isna().any().any():
        raise ValueError(f"{csv_path}: missing (traj, time, cell) samples")
    rsrp = piv.to_numpy(np.float32).reshape(len(trajs), len(times), len(cells))
    pos = (df[df["cell"] == cells[0]].sort_values(["traj", "time_s"])[["x", "y"]]
           .to_numpy(np.float32).reshape(len(trajs), len(times), 2))
    rng = np.random.default_rng(seed)
    speed = df.groupby("traj")["speed_kmh"].first().reindex(trajs).to_numpy(np.float32)
    return {"t_s": times, "pos": pos, "rsrp": rsrp,
            "meas": rsrp + rng.normal(0, meas_noise_db, rsrp.shape).astype(np.float32),
            "speed": speed, "sites": np.zeros((len(cells), 2))}


def l3_filter(meas: np.ndarray, k: int, sample_ms: float, ref_ms: float = 200.0) -> np.ndarray:
    """TS 38.331 5.5.3.2 layer 3 filter F_n = (1-a) F_{n-1} + a M_n, a = 1/2^(k/4), applied in dB.

    The coefficient is specified for a 200 ms measurement period. With 40 ms samples we keep the
    SAME time constant: a' = 1 - (1-a)^(dt/200ms). Without this, k=4 at 40 ms would filter five
    times less than intended and inflate ping-pong for every policy.
    """
    a = 1.0 / 2 ** (k / 4)
    a_eff = 1.0 - (1.0 - a) ** (sample_ms / ref_ms)
    out = np.empty_like(meas)
    out[..., 0, :] = meas[..., 0, :]
    for t in range(1, meas.shape[-2]):
        out[..., t, :] = (1 - a_eff) * out[..., t - 1, :] + a_eff * meas[..., t, :]
    return out
