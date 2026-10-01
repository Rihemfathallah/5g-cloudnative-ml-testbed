"""Fault and traffic plans.

Plans are drawn up front from a seed, so a run is reproducible and the plan is inspectable before
anything is injected (`tb plan configs/runs/x.yaml`). Execution then follows the plan but always
measures real applied/removed times -- the plan never becomes ground truth.

Design choices worth stating in the report:
* Fault types are drawn by *stratified* sampling (shuffled rounds over the catalog), and severities
  alternate within each type, so every (type, severity) cell gets support even in a short run.
  i.i.d. sampling over 7 types and ~25 faults per run often leaves a cell with zero or one fault.
* The gap before a fault is measured from the END of the previous effect window
  (t_removed + grace), not from the planned end, because pod restarts take a variable time.
"""
from __future__ import annotations

import dataclasses as dc
import json
from typing import Iterator

import numpy as np

from testbed.common.config import parse_duration, parse_range


@dc.dataclass(frozen=True)
class FaultPlan:
    fault_id: str
    type: str
    severity: str
    params: dict
    target: str
    injector: str
    gap_before_s: float        # wait after the previous effect window ends
    duration_s: float          # how long the impairment is held (pod_kill: settle time after Ready)

    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)


@dc.dataclass(frozen=True)
class TrafficEpisode:
    kind: str                  # iperf_burst | ping_only | idle | reattach
    duration_s: float
    params: dict


def _stratified_types(catalog: list[dict], n: int, rng: np.random.Generator) -> list[int]:
    order: list[int] = []
    while len(order) < n:
        order.extend(rng.permutation(len(catalog)).tolist())
    return order[:n]


def plan_faults(fault_cfg: dict, collection_s: float, seed: int) -> list[FaultPlan]:
    """Draw a fault sequence that fits in `collection_s` under the gap rules.

    The fit uses planned durations; the executor stops scheduling new faults once the remaining
    time cannot hold a fault plus its gap, so a slow pod restart never pushes a fault past the end.
    """
    rng = np.random.default_rng(seed)
    catalog = fault_cfg["catalog"]
    min_gap = parse_duration(fault_cfg["min_gap"])
    first_after = parse_duration(fault_cfg.get("first_after", fault_cfg["min_gap"]))
    grace = parse_duration(fault_cfg.get("grace", "60s"))
    # Upper bound on how many faults can fit, then stratify over that many.
    shortest = min(parse_range(c["duration"])[0] for c in catalog)
    n_max = int(collection_s // (shortest + grace + min_gap)) + 1
    type_order = _stratified_types(catalog, n_max, rng)
    sev_counter: dict[int, int] = {}
    sev_offset = {i: int(rng.integers(2)) for i in range(len(catalog))}

    plans: list[FaultPlan] = []
    t = 0.0
    for k, ci in enumerate(type_order):
        c = catalog[ci]
        gap = first_after if k == 0 else min_gap + float(rng.uniform(0.0, min_gap))
        lo, hi = parse_range(c["duration"])
        dur = float(rng.uniform(lo, hi))
        if t + gap + dur + grace > collection_s:
            break
        j = sev_counter.get(ci, 0)
        sev_counter[ci] = j + 1
        severity = ("mild", "strong")[(j + sev_offset[ci]) % 2]
        target = c["targets"][int(rng.integers(len(c["targets"])))]
        plans.append(FaultPlan(
            fault_id=f"f{len(plans):03d}", type=c["type"], severity=severity,
            params=dict(c.get(severity, {})), target=target, injector=c["injector"],
            gap_before_s=gap, duration_s=dur,
        ))
        t += gap + dur + grace
    return plans


def iter_traffic(traffic_cfg: dict, seed: int) -> Iterator[TrafficEpisode]:
    """Endless random sequence of traffic episodes (the caller stops it)."""
    rng = np.random.default_rng(seed)
    kinds = list(traffic_cfg["mix"].keys())
    probs = np.array([traffic_cfg["mix"][k] for k in kinds], dtype=float)
    probs /= probs.sum()
    while True:
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        spec = traffic_cfg.get(kind, {})
        if kind == "reattach":
            lo, hi = parse_range(spec.get("down", ["5s", "20s"]))
            yield TrafficEpisode(kind, float(rng.uniform(lo, hi)), {})
            continue
        lo, hi = parse_range(spec.get("duration", ["30s", "120s"]))
        params: dict = {}
        if kind == "iperf_burst":
            mlo, mhi = spec.get("mbps", [2, 20])
            plo, phi = spec.get("parallel", [1, 1])
            params = {"mbps": round(float(rng.uniform(mlo, mhi)), 2),
                      "parallel": int(rng.integers(plo, phi + 1)),
                      "reverse": bool(rng.integers(2))}   # downlink half the time
        yield TrafficEpisode(kind, float(rng.uniform(lo, hi)), params)
