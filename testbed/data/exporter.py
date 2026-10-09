"""Prometheus -> raw.parquet (canonical long format).

Details that matter:
* Prometheus answers at most 11,000 points per series per query_range call; at a 5 s step that is
  ~15 h, so we query in chunks (default 6 h).
* query_range is inclusive at BOTH ends; consecutive chunks are therefore made disjoint
  (next chunk starts one step after the previous end) to avoid duplicate timestamps.
* `start` is floored to a multiple of the step so every run shares one global time grid.
* An expression that returns no data is reported, never silently dropped; an expression that
  returns more than one series is an error in the metric map (it must aggregate).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from testbed.common.schema import validate_raw
from testbed.data.metric_map import Query, load_metric_map

log = logging.getLogger(__name__)

MAX_POINTS = 11000


class ExportError(RuntimeError):
    pass


def chunk_bounds(start_s: float, end_s: float, step_s: int, chunk_s: float) -> list[tuple[int, int]]:
    """Disjoint, step-aligned [start, end] pairs (seconds, inclusive) covering [start_s, end_s]."""
    if chunk_s / step_s + 1 > MAX_POINTS:
        raise ExportError(f"chunk of {chunk_s}s at step {step_s}s exceeds {MAX_POINTS} points")
    s = int(math.floor(start_s / step_s) * step_s)
    e_final = int(math.floor(end_s / step_s) * step_s)
    out = []
    while s <= e_final:
        e = min(s + int(chunk_s // step_s) * step_s - step_s, e_final)
        out.append((s, e))
        s = e + step_s
    return out


def query_range(base_url: str, q: str, start: int, end: int, step: int,
                session: requests.Session | None = None, timeout: float = 120) -> list[dict]:
    sess = session or requests
    r = sess.get(f"{base_url.rstrip('/')}/api/v1/query_range",
                 params={"query": q, "start": start, "end": end, "step": step}, timeout=timeout)
    if r.status_code != 200:
        raise ExportError(f"HTTP {r.status_code} for {q!r}: {r.text[:300]}")
    body = r.json()
    if body.get("status") != "success":
        raise ExportError(f"query failed {q!r}: {body.get('error')}")
    return body["data"]["result"]


def export_run(base_url: str, metric_map_path, run_id: str, source: str, start_ms: int,
               end_ms: int, out_dir: Path, chunk_s: float = 6 * 3600,
               session: requests.Session | None = None) -> pd.DataFrame:
    queries, meta = load_metric_map(metric_map_path)
    step = int(meta.get("step_s", 5))
    bounds = chunk_bounds(start_ms / 1000, end_ms / 1000, step, chunk_s)
    frames, report = [], {"run_id": run_id, "step_s": step, "chunks": len(bounds), "series": []}
    for qd in queries:
        ts_all, val_all, n_series = [], [], 0
        for (s, e) in bounds:
            res = query_range(base_url, qd.promql, s, e, step, session)
            if len(res) > 1:
                raise ExportError(f"{qd.entity}.{qd.metric}: expression returned {len(res)} series; "
                                  "aggregate it in the metric map")
            if res:
                n_series = 1
                vals = res[0]["values"]
                ts_all.extend(int(round(float(t) * 1000)) for t, _ in vals)
                val_all.extend(float(v) for _, v in vals)  # "NaN" -> nan
        entry = {"entity": qd.entity, "metric": qd.metric, "points": len(ts_all),
                 "verify_name": qd.verify}
        if n_series == 0:
            entry["status"] = "no_data"
            log.warning("no data for %s.%s  (%s)", qd.entity, qd.metric, qd.promql)
        else:
            entry["status"] = "ok"
            frames.append(pd.DataFrame({"ts": ts_all, "value": val_all}).assign(
                entity=qd.entity, metric=qd.metric))
        report["series"].append(entry)
    if not frames:
        raise ExportError("no series returned any data -- is Prometheus scraping the testbed?")
    df = pd.concat(frames, ignore_index=True)
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
    df = df.assign(run_id=run_id, source=source)
    df = validate_raw(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "raw.parquet", index=False)
    with open(out_dir / "export_report.json", "w") as f:
        json.dump(report, f, indent=2)
    n_missing = sum(1 for s in report["series"] if s["status"] != "ok")
    log.info("exported %d rows, %d series, %d without data", len(df),
             len(report["series"]) - n_missing, n_missing)
    return df
