"""Exporter, labels, windows and splits: where leakage and label errors would come from."""
import numpy as np
import pandas as pd
import pytest
import yaml

from testbed.common.schema import SchemaError, validate_raw
from testbed.data.exporter import MAX_POINTS, ExportError, chunk_bounds, export_run
from testbed.data.features import assign_split, build_windows

T0 = 1_800_000_000_000
DET = yaml.safe_load(open("configs/detection.yaml"))


# ------------------------------------------------------------------ exporter
def test_chunks_are_disjoint_aligned_and_cover():
    b = chunk_bounds(1000.7, 1000.7 + 20 * 3600, 5, 6 * 3600)
    assert b[0][0] % 5 == 0
    for (s1, e1), (s2, _) in zip(b, b[1:]):
        assert s2 == e1 + 5                                   # disjoint, no gap
    assert all((e - s) / 5 + 1 <= MAX_POINTS for s, e in b)
    assert b[-1][1] >= 1000.7 + 20 * 3600 - 5
    with pytest.raises(ExportError):
        chunk_bounds(0, 100, 5, 5 * MAX_POINTS * 2)


class FakeResp:
    def __init__(self, body, code=200):
        self.body, self.status_code, self.text = body, code, ""

    def json(self):
        return self.body


class FakeSession:
    """Prometheus stand-in: every query returns a ramp over the requested range."""
    def __init__(self, multi=None, empty=None):
        self.multi, self.empty = multi or set(), empty or set()

    def get(self, url, params, timeout):
        q, s, e, st = params["query"], params["start"], params["end"], params["step"]
        if any(k in q for k in self.empty):
            return FakeResp({"status": "success", "data": {"result": []}})
        vals = [[t, str(t % 97)] for t in range(s, e + 1, st)]
        series = [{"metric": {}, "values": vals}]
        if any(k in q for k in self.multi):
            series = series * 2
        return FakeResp({"status": "success", "data": {"result": series}})


def test_export_long_format(tmp_path):
    start = T0
    df = export_run("http://x", "configs/metric_map.yaml", "r1", "5g", start, start + 3 * 3600_000,
                    tmp_path, chunk_s=3600, session=FakeSession(empty={"ran_ue"}))
    assert set(df.columns) == {"run_id", "ts", "source", "entity", "metric", "value"}
    assert not df.duplicated(["ts", "entity", "metric"]).any()          # chunk boundaries
    assert (df["ts"] % 5000 == 0).all()
    rep = (tmp_path / "export_report.json").read_text()
    assert '"no_data"' in rep and "ran_ue" in rep


def test_export_rejects_unaggregated(tmp_path):
    with pytest.raises(ExportError, match="series"):
        export_run("http://x", "configs/metric_map.yaml", "r1", "5g", T0, T0 + 600_000, tmp_path,
                   session=FakeSession(multi={"container_cpu"}))


def test_raw_schema_catches_seconds():
    df = pd.DataFrame({"run_id": ["r"], "ts": [1_700_000_000], "source": ["s"], "entity": ["e"],
                       "metric": ["m"], "value": [1.0]})
    with pytest.raises(SchemaError, match="milliseconds"):
        validate_raw(df)


# ------------------------------------------------------------------ windows
def _raw(n=2000, cols=("a.x", "b.y")):
    t = T0 + np.arange(n) * 5000
    rng = np.random.default_rng(0)
    frames = [pd.DataFrame({"ts": t, "value": rng.normal(size=n), "entity": c.split(".")[0],
                            "metric": c.split(".")[1]}) for c in cols]
    return pd.concat(frames).assign(run_id="r1", source="t")


LABEL_COLS = ["fault_id", "type", "severity", "params", "target", "injector", "t_applied", "t_removed", "status"]


def _labels(rows):
    return pd.DataFrame(columns=LABEL_COLS, data=[{"fault_id": f"f{i}", "type": "cpu", "severity": "mild", "params": "{}",
                          "target": "pod:upf", "injector": "chaos-mesh", "t_applied": a, "t_removed": b,
                          "status": st} for i, (a, b, st) in enumerate(rows)]) if rows else \
        pd.DataFrame(columns=LABEL_COLS)


def _build(tmp_path, labels, raw=None, det=None):
    d = tmp_path / "r1"
    d.mkdir(exist_ok=True)
    (raw if raw is not None else _raw()).to_parquet(d / "raw.parquet")
    labels.to_csv(d / "labels.csv", index=False)
    return build_windows([d], det or DET, "testbed")


def test_label_rules(tmp_path):
    # fault from sample 100 to 140 (+ 60 s grace = 12 samples) -> effect on samples [100, 152)
    a, b = T0 + 100 * 5000, T0 + 140 * 5000
    ws = _build(tmp_path, _labels([(a, b, "ok")]))
    m = ws.meta.set_index("start_ts")
    s = lambda i: T0 + i * 5000  # noqa: E731
    assert m.loc[s(100), "label"] == 1                    # fully inside
    assert m.loc[s(94), "label"] == 1                     # 6/12 = 50 % -> anomalous
    assert m.loc[s(95), "label"] == 1
    assert m.loc[s(93), "label"] == -1                    # 5/12 -> ambiguous
    assert m.loc[s(88), "label"] == 0                     # touches nothing
    assert m.loc[s(152), "label"] == 0                    # grace ends at sample 152
    assert m.loc[s(141), "label"] == 1                    # inside the grace period
    assert m.loc[s(100), "fault_id"] == "f0"


def test_failed_injection_is_ambiguous(tmp_path):
    a, b = T0 + 100 * 5000, T0 + 140 * 5000
    ws = _build(tmp_path, _labels([(a, b, "failed")]))
    m = ws.meta.set_index("start_ts")
    assert m.loc[T0 + 110 * 5000, "label"] == -1


def test_splits_do_not_share_samples(tmp_path):
    ws = _build(tmp_path, _labels([]))
    m = ws.meta
    tr_end = m[m.split == "train"]["end_ts"].max()
    va_start, va_end = m[m.split == "val"]["start_ts"].min(), m[m.split == "val"]["end_ts"].max()
    te_start = m[m.split == "test"]["start_ts"].min()
    assert va_start >= tr_end + 60_000 and te_start >= va_end + 60_000      # purge
    assert set(m["split"]) == {"train", "val", "test"}


def test_assign_split_fractions():
    start = np.arange(0, 1000, 10) * 1000
    split, (b1, b2) = assign_split(start, start + 60_000, 0, 1_060_000, [0.6, 0.2, 0.2], 60_000)
    assert b1 == 0.6 * 1_060_000 and b2 == 0.8 * 1_060_000
    assert (split[start + 60_000 <= b1] == "train").all()


def test_missing_and_constant_columns(tmp_path):
    raw = _raw(cols=("a.x", "b.y", "c.const", "d.gappy"))
    raw.loc[raw.metric == "const", "value"] = 1.0
    gappy = raw.metric == "gappy"
    raw = raw[~gappy | (np.arange(len(raw)) % 3 == 0)]                    # 2/3 missing
    det = yaml.safe_load(open("configs/detection.yaml"))
    det["windowing"]["testbed"]["ffill_max_s"] = 0
    ws = _build(tmp_path, _labels([]), raw, det)
    assert ws.report["dropped_columns"] == {"c.const": "constant_in_train", "d.gappy": "too_missing"}
    assert "a.x|missing" in ws.feature_names and not np.isnan(ws.X).any()


def test_exclude_columns(tmp_path):
    det = yaml.safe_load(open("configs/detection.yaml"))
    det["columns"]["exclude"] = [r"^b\."]
    ws = _build(tmp_path, _labels([]), det=det)
    assert ws.columns == ["a.x"]


def test_test_only_runs(tmp_path):
    for name in ("r1", "r2"):
        d = tmp_path / name
        d.mkdir()
        _raw().assign(run_id=name).to_parquet(d / "raw.parquet")
        _labels([]).to_csv(d / "labels.csv", index=False)
    ws = build_windows([tmp_path / "r1", tmp_path / "r2"], DET, "testbed", test_only={"r2"})
    assert set(ws.meta[ws.meta.run_id == "r2"]["split"]) == {"test"}
    assert "train" in set(ws.meta[ws.meta.run_id == "r1"]["split"])
