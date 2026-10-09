"""Detection experiment: windows -> every model x every seed -> results/<name>/*.csv.

Outputs (all CSV, one row per model/seed/... so figure scripts only read and plot):
  window_metrics.csv     precision/recall/F1 on test windows
  events.csv             one row per test fault and model/seed: detected, delay
  event_summary.csv      event recall and delays per (model, seed, fault_type, severity)
  false_alarms.csv       false alarms per hour of normal operation
  window_recall_type.csv window-level recall per fault type
  attribution.csv        LSTM-AE: share of reconstruction error per column, by fault type
  thresholds.json        chosen thresholds and the rule that chose them
  summary.csv            mean +- std over seeds of the headline numbers
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from testbed.anomaly.detectors import DETERMINISTIC, REGISTRY
from testbed.anomaly.evaluate import (event_detection, false_alarms_per_hour, summarize_events,
                                      window_metrics, window_recall_by_type)
from testbed.common.runinfo import seed_everything
from testbed.data.features import WindowSet

log = logging.getLogger(__name__)


def load_faults(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        p = Path(d) / "labels.csv"
        if p.exists():
            df = pd.read_csv(p)
            run_id = pd.read_parquet(Path(d) / "raw.parquet", columns=["run_id"])["run_id"].iloc[0] \
                if (Path(d) / "raw.parquet").exists() else Path(d).name
            frames.append(df.assign(run_id=run_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_detection(ws: WindowSet, faults: pd.DataFrame, det_cfg: dict, out_dir: Path,
                  models: list[str] | None = None, seeds: list[int] | None = None,
                  tag: dict | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = models or ["zscore", "iforest", "lstm_ae", "rf"]
    seeds = seeds if seeds is not None else det_cfg["evaluation"]["seeds"]
    grace = float(det_cfg["labels"]["grace_s"])
    stride_s = ws.report["step_s"] * ws.report["stride"]
    merge_s = det_cfg["evaluation"]["alarm_merge_s"]
    split = ws.meta["split"].to_numpy()
    train, val, test = ws.subset(split == "train"), ws.subset(split == "val"), ws.subset(split == "test")
    if len(test) == 0:
        raise ValueError("no test windows")
    log.info("train %d (normal %d) / val %d / test %d windows", len(train),
             int((train.meta.label == 0).sum()), len(val), len(test))

    wm_rows, ev_frames, fa_rows, wr_frames, attr_frames, thr = [], [], [], [], [], {}
    for mname in models:
        mcfg = dict(det_cfg["models"]["rf" if mname == "rf" else mname])
        mcfg.update(_stride_s=stride_s, _merge_s=merge_s)
        for seed in (seeds[:1] if mname in DETERMINISTIC else seeds):
            seed_everything(seed)
            t0 = time.time()
            det = REGISTRY[mname](mcfg, seed).fit(train, val)
            fit_s = time.time() - t0
            pred = det.predict(test)
            key = {"model": det.name, "seed": seed, **(tag or {})}
            wm_rows.append({**key, **window_metrics(pred, test.meta["label"].to_numpy()),
                            "fit_seconds": round(fit_s, 2)})
            fa, normal_h = false_alarms_per_hour(pred, test.meta, stride_s, merge_s)
            fa_rows.append({**key, "false_alarms_per_h": fa, "normal_hours": normal_h})
            ev = event_detection(pred, test.meta, faults, grace)
            if len(ev):
                ev_frames.append(ev.assign(**key))
            wr = window_recall_by_type(pred, test.meta)
            if len(wr):
                wr_frames.append(wr.assign(**key))
            thr[f"{det.name}/seed{seed}"] = {"threshold": float(det.threshold),
                                             "rule": mcfg.get("threshold_rule", "fixed/quantile")}
            if mname == "lstm_ae":
                thr[f"{det.name}/seed{seed}"]["history"] = det.history
                attr = det.explain(test)
                hit = pred & (test.meta["label"] == 1).to_numpy()
                if hit.any():
                    a = attr[hit].assign(fault_type=test.meta["fault_type"].to_numpy()[hit])
                    long = a.melt(id_vars="fault_type", var_name="column", value_name="share")
                    attr_frames.append(long.groupby(["fault_type", "column"])["share"].mean()
                                       .reset_index().assign(**key))
            log.info("%-14s seed %d  F1 %.3f  FA/h %.2f  (%.1fs)", det.name, seed,
                     wm_rows[-1]["f1"], fa, fit_s)

    wm = pd.DataFrame(wm_rows)
    wm.to_csv(out_dir / "window_metrics.csv", index=False)
    pd.DataFrame(fa_rows).to_csv(out_dir / "false_alarms.csv", index=False)
    events = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
    events.to_csv(out_dir / "events.csv", index=False)
    summ = summarize_events(events, ["model", "seed", "type", "severity"]) if len(events) else pd.DataFrame()
    summ.to_csv(out_dir / "event_summary.csv", index=False)
    (pd.concat(wr_frames) if wr_frames else pd.DataFrame()).to_csv(out_dir / "window_recall_type.csv", index=False)
    (pd.concat(attr_frames) if attr_frames else pd.DataFrame()).to_csv(out_dir / "attribution.csv", index=False)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thr, f, indent=2)

    summary = headline(wm, pd.DataFrame(fa_rows), events)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def headline(wm: pd.DataFrame, fa: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    per_seed = wm[["model", "seed", "f1", "precision", "recall"]].merge(
        fa[["model", "seed", "false_alarms_per_h"]], on=["model", "seed"])
    if len(events):
        e = events[~events["excluded"].astype(bool)]
        ev = e.groupby(["model", "seed"]).agg(event_recall=("detected", "mean"),
                                              delay_median_s=("delay_s", "median"),
                                              n_test_faults=("fault_id", "size")).reset_index()
        per_seed = per_seed.merge(ev, on=["model", "seed"], how="left")
    agg = per_seed.drop(columns="seed").groupby("model")
    out = agg.mean().add_suffix("_mean").join(agg.std(ddof=0).add_suffix("_std"))
    out["n_seeds"] = agg.size()
    return out.reset_index()
