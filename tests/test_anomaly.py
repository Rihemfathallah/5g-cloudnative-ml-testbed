"""Evaluation protocol and detectors."""
import numpy as np
import pandas as pd
import pytest

from testbed.anomaly.detectors import best_f1_threshold
from testbed.anomaly.evaluate import (alarm_events, event_detection, false_alarms_per_hour,
                                      window_metrics)

T0 = 1_800_000_000_000


def meta(n, labels=None, fault=None, split="test"):
    start = T0 + np.arange(n) * 5000
    m = pd.DataFrame({"run_id": "r", "start_ts": start, "end_ts": start + 60_000,
                      "label": labels if labels is not None else np.zeros(n, int),
                      "fault_id": fault if fault is not None else [""] * n, "split": split})
    return m


def test_window_metrics_excludes_ambiguous():
    m = window_metrics(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 1, 0, -1]))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_alarm_merging_and_false_alarms():
    mm = meta(100)
    pred = np.zeros(100, bool)
    pred[[10, 11, 12, 13]] = True      # consecutive (5 s apart) -> one event
    pred[[40, 47]] = True              # 35 s apart > 30 s merge -> two events
    ev = alarm_events(pred, mm, merge_s=30)
    assert len(ev) == 3
    fa, h = false_alarms_per_hour(pred, mm, stride_s=5, merge_s=30)
    assert h == pytest.approx(100 * 5 / 3600)
    assert fa == pytest.approx(3 / h)


def test_event_detection_delay():
    n = 400
    a = T0 + 200 * 5000                                   # fault applied at sample 200
    faults = pd.DataFrame([{"run_id": "r", "fault_id": "f0", "type": "cpu", "severity": "mild",
                            "t_applied": a, "t_removed": a + 120_000, "status": "ok"}])
    mm = meta(n)
    pred = np.zeros(n, bool)
    pred[195] = True        # window [195, 207) overlaps the effect window: ends 35 s after a
    det = event_detection(pred, mm, faults, grace_s=60)
    assert bool(det.iloc[0]["detected"]) and det.iloc[0]["delay_s"] == pytest.approx(35.0)
    # an alarm that ends before the fault starts does not count
    pred2 = np.zeros(n, bool)
    pred2[180] = True       # ends at sample 192 < 200
    assert not bool(event_detection(pred2, mm, faults, 60).iloc[0]["detected"])


def test_fault_crossing_split_is_excluded():
    mm = meta(400)
    mm.loc[:100, "split"] = "val"
    a = T0 + 90 * 5000                    # starts before the first test window
    faults = pd.DataFrame([{"run_id": "r", "fault_id": "f0", "type": "cpu", "severity": "mild",
                            "t_applied": a, "t_removed": a + 60_000, "status": "ok"}])
    det = event_detection(np.ones(400, bool), mm[mm.split == "test"], faults, 60)
    assert bool(det.iloc[0]["excluded"])


def test_best_f1_threshold():
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    thr = best_f1_threshold(s, y)
    assert 0.3 < thr < 0.8
    assert ((s > thr) == (y == 1)).all()


@pytest.mark.slow
def test_detectors_separate_easy_data(tmp_path):
    """On data with an obvious shift, every unsupervised model must beat chance by far."""
    import yaml
    from testbed.anomaly.detectors import REGISTRY
    from testbed.data.features import build_windows
    n = 3000
    t = T0 + np.arange(n) * 5000
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    a, b = 2000, 2060
    x[a:b] += 8.0
    raw = pd.DataFrame({"ts": t, "value": x, "entity": "e", "metric": "m", "run_id": "r1", "source": "t"})
    raw2 = raw.assign(entity="f", value=rng.normal(size=n))
    d = tmp_path / "r1"
    d.mkdir()
    pd.concat([raw, raw2]).to_parquet(d / "raw.parquet")
    pd.DataFrame([{"fault_id": "f0", "type": "cpu", "severity": "strong", "params": "{}", "target": "pod:upf",
                   "injector": "chaos-mesh", "t_applied": int(t[a]), "t_removed": int(t[b]), "status": "ok"}]
                 ).to_csv(d / "labels.csv", index=False)
    det = yaml.safe_load(open("configs/detection.yaml"))
    det["split"]["fractions"] = [0.5, 0.15, 0.35]
    ws = build_windows([d], det, "testbed")
    det["models"]["lstm_ae"]["epochs"] = 8
    sp = ws.meta["split"].to_numpy()
    tr, va, te = ws.subset(sp == "train"), ws.subset(sp == "val"), ws.subset(sp == "test")
    y = te.meta["label"].to_numpy()
    for name in ("zscore", "iforest", "lstm_ae"):
        cfg = dict(det["models"][name])
        cfg.update(_stride_s=5, _merge_s=30)
        model = REGISTRY[name](cfg, 0).fit(tr, va)
        s = model.score(te)
        assert s[y == 1].mean() > s[y == 0].mean(), name
