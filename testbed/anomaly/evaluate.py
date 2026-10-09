"""Evaluation protocol (Section 6 of the architecture doc).

* window level   precision / recall / F1, ambiguous windows excluded
* event level    a fault is detected if ANY alarm window overlaps its effect window;
                 delay = first such alarm's window END (the moment the decision exists) - t_applied
* false alarms   alarms on purely normal windows, merged within `merge_s` into alarm events,
                 per hour of normal operation
* no point-adjustment: it is known to inflate scores (Kim et al., AAAI 2022) and is not used.

Only faults whose whole effect window lies inside the test portion of their run are scored at
event level; faults cut by a split boundary are reported as `excluded`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_metrics(pred: np.ndarray, labels: np.ndarray) -> dict:
    m = labels >= 0
    y, p = labels[m] == 1, pred[m]
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def alarm_events(pred: np.ndarray, meta: pd.DataFrame, merge_s: float) -> pd.DataFrame:
    rows = []
    merge_ms = merge_s * 1000
    for run_id, g in meta.assign(pred=pred).groupby("run_id", sort=False):
        g = g.sort_values("end_ts")
        a = g[g["pred"]]
        if a.empty:
            continue
        ev_id = (a["end_ts"].diff().fillna(np.inf) > merge_ms).cumsum()
        for _, e in a.groupby(ev_id):
            rows.append({"run_id": run_id, "t_first": int(e["end_ts"].min()),
                         "t_last": int(e["end_ts"].max()), "n_windows": len(e),
                         "all_normal": bool((e["label"] == 0).all() and (e["fault_id"] == "").all())})
    return pd.DataFrame(rows, columns=["run_id", "t_first", "t_last", "n_windows", "all_normal"])


def false_alarms_per_hour(pred, meta, stride_s, merge_s) -> tuple[float, float]:
    ev = alarm_events(pred, meta, merge_s)
    normal_h = float((meta["label"] == 0).sum()) * stride_s / 3600
    n_fa = int(ev["all_normal"].sum()) if len(ev) else 0
    return (n_fa / normal_h if normal_h > 0 else float("nan")), normal_h


def test_faults(meta: pd.DataFrame, faults: pd.DataFrame, grace_s: float) -> pd.DataFrame:
    """Ok faults whose whole effect window is inside the test span of their run."""
    out = []
    test = meta[meta["split"] == "test"]
    spans = test.groupby("run_id").agg(lo=("start_ts", "min"), hi=("end_ts", "max"))
    for _, f in faults[faults["status"] == "ok"].iterrows():
        if f["run_id"] not in spans.index:
            continue
        lo, hi = spans.loc[f["run_id"], ["lo", "hi"]]
        a, b = int(f["t_applied"]), int(f["t_removed"]) + int(grace_s * 1000)
        out.append({**f.to_dict(), "eff_start": a, "eff_end": b, "in_test": bool(a >= lo and b <= hi)})
    return pd.DataFrame(out)


def event_detection(pred: np.ndarray, meta: pd.DataFrame, faults: pd.DataFrame,
                    grace_s: float) -> pd.DataFrame:
    tf = test_faults(meta, faults, grace_s)
    if tf.empty:
        return tf
    rows = []
    for _, f in tf.iterrows():
        if not f["in_test"]:
            rows.append({**f, "detected": np.nan, "delay_s": np.nan, "excluded": True})
            continue
        g = meta[(meta["run_id"] == f["run_id"]) & pred]
        hit = g[(g["end_ts"] > f["eff_start"]) & (g["start_ts"] < f["eff_end"])]
        det = len(hit) > 0
        delay = (hit["end_ts"].min() - f["eff_start"]) / 1000 if det else np.nan
        rows.append({**f, "detected": det, "delay_s": delay, "excluded": False})
    return pd.DataFrame(rows)


def summarize_events(det: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = det[~det["excluded"].astype(bool)]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(by, dropna=False)
    return pd.DataFrame({
        "n_faults": g.size(),
        "event_recall": g["detected"].mean(),
        "delay_median_s": g["delay_s"].median(),
        "delay_p90_s": g["delay_s"].quantile(0.9),
    }).reset_index()


def window_recall_by_type(pred, meta) -> pd.DataFrame:
    a = meta.assign(pred=pred)
    a = a[a["label"] == 1]
    if a.empty:
        return pd.DataFrame()
    return a.groupby(["fault_type", "severity"])["pred"].mean().rename("window_recall").reset_index()
