"""raw.parquet + labels.csv  ->  windows (features, sequences, labels, split).

Label rules (ADR-6):
  effect window E = [t_applied, t_removed + grace)
  a window is ANOMALOUS (1) if >= 50 % of its samples fall in some E, NORMAL (0) if none does,
  AMBIGUOUS (-1) otherwise. Failed injections make their whole interval ambiguous: we do not know
  what they did.

Split rules:
  per run, by time: [0, 60 %) train, [60, 80 %) validation, [80, 100 %] test. A window belongs to a
  split only if it lies entirely inside it, and validation/test windows must start >= purge_s after
  the boundary, so no sample is shared across splits and neighbouring windows cannot leak.

Missing data:
  a column missing in more than `max_missing_ratio` of TRAINING samples is dropped (reported);
  remaining gaps are forward-filled up to ffill_max_s, then imputed with the training median,
  and each column also yields a missing-fraction feature, so an outage (e.g. no ping reply) stays
  visible to the models instead of being imputed away.
"""
from __future__ import annotations

import dataclasses as dc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from testbed.common.schema import column_name, validate_labels, validate_raw

log = logging.getLogger(__name__)

STATS = ("mean", "std", "min", "max", "delta")


@dc.dataclass
class RunGrid:
    run_id: str
    t: np.ndarray                 # (n,) int64 ms, regular grid
    values: pd.DataFrame          # (n, m) wide, NaN where missing
    labels: pd.DataFrame


@dc.dataclass
class WindowSet:
    meta: pd.DataFrame            # one row per window
    X: np.ndarray                 # (N, F) tabular features
    feature_names: list[str]
    seq: np.ndarray               # (N, W, C) normalised sequences
    seq_channels: list[str]
    columns: list[str]            # value columns kept
    report: dict

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(self.meta[mask].reset_index(drop=True), self.X[mask], self.feature_names,
                         self.seq[mask], self.seq_channels, self.columns, self.report)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.meta.copy()
        feat = pd.DataFrame(self.X, columns=self.feature_names)
        pd.concat([df, feat], axis=1).to_parquet(out_dir / "windows.parquet", index=False)
        np.save(out_dir / "sequences.npy", self.seq)
        with open(out_dir / "windows_report.json", "w") as f:
            json.dump({**self.report, "seq_channels": self.seq_channels, "columns": self.columns},
                      f, indent=2, default=str)

    @classmethod
    def load(cls, out_dir: Path) -> "WindowSet":
        df = pd.read_parquet(out_dir / "windows.parquet")
        rep = json.loads((out_dir / "windows_report.json").read_text())
        meta_cols = rep["meta_columns"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        return cls(df[meta_cols], df[feat_cols].to_numpy(np.float32), feat_cols,
                   np.load(out_dir / "sequences.npy"), rep["seq_channels"], rep["columns"], rep)


# ----------------------------------------------------------------------------------- grid
def to_grid(raw: pd.DataFrame, step_s: int, ffill_max_s: float) -> tuple[np.ndarray, pd.DataFrame]:
    step = step_s * 1000
    raw = raw.assign(col=[column_name(e, m) for e, m in zip(raw["entity"], raw["metric"])],
                     tg=(raw["ts"] // step) * step)
    wide = raw.pivot_table(index="tg", columns="col", values="value", aggfunc="mean")
    t = np.arange(wide.index.min(), wide.index.max() + step, step, dtype=np.int64)
    wide = wide.reindex(t)
    limit = int(ffill_max_s // step_s)
    if limit > 0:
        wide = wide.ffill(limit=limit)
    wide.index.name = "ts"
    return t, wide.sort_index(axis=1)


def effect_intervals(labels: pd.DataFrame, grace_s: float) -> tuple[list, list]:
    ok, amb = [], []
    for _, r in labels.iterrows():
        iv = (int(r["t_applied"]), int(r["t_removed"]) + int(grace_s * 1000), r)
        (ok if r["status"] == "ok" else amb).append(iv)
    return ok, amb


def sample_membership(t: np.ndarray, labels: pd.DataFrame, grace_s: float):
    """Per grid sample: index of the ok fault whose effect window contains it (-1 none), and a
    flag for samples inside a failed injection's interval."""
    ok, amb = effect_intervals(labels, grace_s)
    owner = np.full(len(t), -1, dtype=np.int64)
    for k, (a, b, _) in enumerate(ok):
        owner[(t >= a) & (t < b)] = k
    failed = np.zeros(len(t), bool)
    for a, b, _ in amb:
        failed |= (t >= a) & (t < b)
    return owner, failed, ok


# ----------------------------------------------------------------------------------- windows
def _window_index(n: int, window: int, stride: int) -> np.ndarray:
    if n < window:
        return np.zeros((0, window), dtype=np.int64)
    starts = np.arange(0, n - window + 1, stride)
    return starts[:, None] + np.arange(window)[None, :]


def label_windows(t, idx, owner, failed, ok, step_ms, overlap_thr):
    n_w = len(idx)
    own = owner[idx]                                   # (N, W)
    inside = own >= 0
    frac = inside.mean(axis=1)
    lab = np.where(frac >= overlap_thr, 1, np.where(frac == 0, 0, -1))
    lab = np.where(failed[idx].any(axis=1), -1, lab)
    fault_idx = np.full(n_w, -1)
    for i in np.flatnonzero(inside.any(axis=1)):
        vals, counts = np.unique(own[i][own[i] >= 0], return_counts=True)
        fault_idx[i] = vals[np.argmax(counts)]
    ftype = np.array([ok[k][2]["type"] if k >= 0 else "" for k in fault_idx], dtype=object)
    fsev = np.array([ok[k][2]["severity"] if k >= 0 else "" for k in fault_idx], dtype=object)
    fid = np.array([ok[k][2]["fault_id"] if k >= 0 else "" for k in fault_idx], dtype=object)
    return lab, frac, fid, ftype, fsev


def assign_split(start_ms, end_ms, t0, t1, fractions, purge_ms):
    span = t1 - t0
    b1 = t0 + fractions[0] * span
    b2 = t0 + (fractions[0] + fractions[1]) * span
    split = np.full(len(start_ms), "", dtype=object)
    split[end_ms <= b1] = "train"
    split[(start_ms >= b1 + purge_ms) & (end_ms <= b2)] = "val"
    split[start_ms >= b2 + purge_ms] = "test"
    return split, (b1, b2)


def _window_stats(v: np.ndarray) -> np.ndarray:
    """v: (N, W, m) with NaN -> (N, m*6) features: 5 stats + missing fraction per column."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(v, axis=1)
            std = np.nanstd(v, axis=1)
            mn = np.nanmin(v, axis=1)
            mx = np.nanmax(v, axis=1)
        # delta = last valid - first valid
        valid = ~np.isnan(v)
        first_i = valid.argmax(axis=1)
        last_i = v.shape[1] - 1 - valid[:, ::-1, :].argmax(axis=1)
        rows = np.arange(v.shape[0])[:, None]
        cols = np.arange(v.shape[2])[None, :]
        delta = v[rows, last_i, cols] - v[rows, first_i, cols]
        delta[~valid.any(axis=1)] = np.nan
        miss = 1.0 - valid.mean(axis=1)
    return np.concatenate([mean, std, mn, mx, delta, miss], axis=1)


def build_windows(run_dirs: list[Path], det_cfg: dict, profile: str = "testbed",
                  raw_override: dict | None = None,
                  test_only: set[str] | None = None) -> WindowSet:
    """`test_only`: run directory names whose windows all go to the test split (transfer
    experiments, e.g. train on srsRAN runs, test on an OAI run). They never influence the column
    policy, imputation or normalisation, which are fitted on training windows only."""
    test_only = set(test_only or ())
    wcfg = det_cfg["windowing"][profile]
    step_s, W, stride = int(wcfg["step_s"]), int(wcfg["window"]), int(wcfg["stride"])
    grace_s = float(det_cfg["labels"]["grace_s"])
    overlap = float(det_cfg["labels"]["anomalous_overlap"])
    fractions = det_cfg["split"]["fractions"]
    purge_ms = int(det_cfg["split"]["purge_s"] * 1000)

    grids: list[RunGrid] = []
    for d in run_dirs:
        d = Path(d)
        raw = (raw_override or {}).get(d.name)
        if raw is None:
            raw = pd.read_parquet(d / "raw.parquet")
        raw = validate_raw(raw)
        lab_path = d / "labels.csv"
        labels = validate_labels(pd.read_csv(lab_path)) if lab_path.exists() else pd.DataFrame(
            columns=["fault_id", "type", "severity", "params", "target", "injector",
                     "t_applied", "t_removed", "status"])
        t, wide = to_grid(raw, step_s, wcfg["ffill_max_s"])
        g = RunGrid(raw["run_id"].iloc[0], t, wide, labels)
        g.test_only = d.name in test_only  # type: ignore[attr-defined]
        grids.append(g)

    import re
    excl = [re.compile(r) for r in det_cfg.get("columns", {}).get("exclude", [])]
    all_cols = sorted(c for c in set().union(*[set(g.values.columns) for g in grids])
                      if not any(r.search(c) for r in excl))
    metas, values, report = [], [], {"runs": [], "dropped_columns": {}}
    for g in grids:
        vals = g.values.reindex(columns=all_cols)
        idx = _window_index(len(g.t), W, stride)
        owner, failed, ok = sample_membership(g.t, g.labels, grace_s)
        lab, frac, fid, ftype, fsev = label_windows(g.t, idx, owner, failed, ok, step_s * 1000, overlap)
        start = g.t[idx[:, 0]]
        end = g.t[idx[:, -1]] + step_s * 1000
        split, (b1, b2) = assign_split(start, end, g.t[0], g.t[-1] + step_s * 1000, fractions, purge_ms)
        if getattr(g, "test_only", False):
            split[:] = "test"
            b1 = b2 = g.t[0]
        metas.append(pd.DataFrame({
            "run_id": g.run_id, "start_ts": start, "end_ts": end, "label": lab, "overlap": frac,
            "fault_id": fid, "fault_type": ftype, "severity": fsev, "split": split}))
        values.append((vals.to_numpy(np.float64), idx))
        report["runs"].append({"run_id": g.run_id, "n_samples": len(g.t), "n_windows": len(idx),
                               "split_bounds_ms": [int(b1), int(b2)],
                               "n_faults_ok": int((g.labels["status"] == "ok").sum()) if len(g.labels) else 0})

    meta = pd.concat(metas, ignore_index=True)
    # ---- column policy on TRAINING samples only
    train_rows = []
    for (v, idx), m in zip(values, metas):
        tr = idx[(m["split"] == "train").to_numpy()]
        if len(tr):
            train_rows.append(v[np.unique(tr.ravel())])
    train_vals = np.concatenate(train_rows) if train_rows else np.zeros((0, len(all_cols)))
    if len(train_vals) == 0:
        raise ValueError("no training windows: runs too short for the split")
    miss_ratio = np.isnan(train_vals).mean(axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_std = np.nanstd(train_vals, axis=0)
    keep = (miss_ratio <= det_cfg["missing"]["max_missing_ratio"]) & (np.nan_to_num(col_std) > 0)
    for c, mr, s, k in zip(all_cols, miss_ratio, col_std, keep):
        if not k:
            report["dropped_columns"][c] = ("too_missing" if mr > det_cfg["missing"]["max_missing_ratio"]
                                            else "constant_in_train")
    cols = [c for c, k in zip(all_cols, keep) if k]
    if not cols:
        raise ValueError("every column was dropped; check the metric map / export report")
    keep_idx = np.flatnonzero(keep)

    # ---- tabular features
    feats = []
    for v, idx in values:
        feats.append(_window_stats(v[:, keep_idx][idx]))
    X = np.concatenate(feats).astype(np.float64)
    fnames = [f"{c}|{s}" for s in (*STATS, "missing") for c in cols]
    tr_mask = (meta["split"] == "train").to_numpy()
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X[tr_mask], axis=0)
    med = np.nan_to_num(med)
    nan_pos = np.isnan(X)
    X[nan_pos] = np.take(med, np.nonzero(nan_pos)[1])
    if det_cfg["missing"].get("add_missing_feature", True) is False:
        sel = [i for i, n in enumerate(fnames) if not n.endswith("|missing")]
        X, fnames = X[:, sel], [fnames[i] for i in sel]

    # ---- sequences: impute with train median, z-normalise with train-normal stats,
    #      plus a mask channel for every column that is ever missing in training
    train_norm = tr_mask & (meta["label"] == 0).to_numpy()
    tv = train_vals[:, keep_idx]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmedian(tv, axis=0)
            q75, q25 = np.nanpercentile(tv, 75, axis=0), np.nanpercentile(tv, 25, axis=0)
            sd = np.nanstd(tv, axis=0)
    scale = np.where((q75 - q25) > 1e-12, (q75 - q25) / 1.349, np.where(sd > 1e-12, sd, 1.0))
    masked_cols = [i for i in range(len(cols)) if np.isnan(tv[:, i]).any()]
    seqs = []
    for v, idx in values:
        vv = v[:, keep_idx]
        miss = np.isnan(vv)
        z = (np.where(miss, mu, vv) - mu) / scale
        z = np.clip(z, -50, 50)   # a 100x outlier must not dominate the AE loss numerically
        chan = np.concatenate([z, miss[:, masked_cols].astype(float)], axis=1)
        seqs.append(chan[idx])
    seq = np.concatenate(seqs).astype(np.float32)
    seq_channels = cols + [f"{cols[i]}|missing" for i in masked_cols]

    report.update({
        "profile": profile, "step_s": step_s, "window": W, "stride": stride,
        "n_windows": int(len(meta)), "n_train_normal": int(train_norm.sum()),
        "label_counts": {str(k): int(v) for k, v in meta["label"].value_counts().items()},
        "split_counts": {str(k): int(v) for k, v in meta["split"].value_counts().items()},
        "columns_kept": len(cols), "meta_columns": list(meta.columns),
        "excluded_patterns": det_cfg.get("columns", {}).get("exclude", []),
        "norm": {"median": dict(zip(cols, map(float, mu))), "scale": dict(zip(cols, map(float, scale)))},
    })
    log.info("windows: %d (train %d, val %d, test %d); columns kept %d, dropped %d",
             len(meta), (meta.split == "train").sum(), (meta.split == "val").sum(),
             (meta.split == "test").sum(), len(cols), len(report["dropped_columns"]))
    keep_rows = (meta["split"] != "").to_numpy()   # windows straddling a boundary are discarded
    ws = WindowSet(meta, X.astype(np.float32), fnames, seq, seq_channels, cols, report)
    return ws.subset(keep_rows)
