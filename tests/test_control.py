"""Scheduling, execution and injector rendering -- the code that produces ground truth."""
import json
import socket
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml

from testbed.common.clock import SimClock
from testbed.common.config import parse_duration, parse_range, parse_size_bytes
from testbed.common.schema import validate_labels
from testbed.control.injectors import (InjectionError, MininetInjector, MininetRPC, SimInjector,
                                       chaos_manifest)
from testbed.control.runner import FaultRunner, LabelWriter
from testbed.control.schedule import FaultPlan, iter_traffic, plan_faults

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/runs/k8s_srsran_mixed_a.yaml").read_text())


def test_parsers():
    assert parse_duration("15m") == 900 and parse_duration("200ms") == 0.2 and parse_duration(3) == 3.0
    assert parse_range(["2m", "10m"]) == (120.0, 600.0)
    assert parse_size_bytes("256MB") == 256 * 1024**2
    with pytest.raises(ValueError):
        parse_duration("5 minutes")
    with pytest.raises(ValueError):
        parse_range(["10m", "2m"])


def test_plan_is_deterministic_and_fits():
    a = plan_faults(CFG["faults"], 5 * 3600, seed=1)
    b = plan_faults(CFG["faults"], 5 * 3600, seed=1)
    assert a == b and len(a) > 0
    total = sum(p.gap_before_s + p.duration_s + 60 for p in a)
    assert total <= 5 * 3600
    assert a[0].gap_before_s == parse_duration(CFG["faults"]["first_after"])
    assert all(p.gap_before_s >= 15 * 60 for p in a[1:])


def test_plan_is_stratified():
    """In any window of len(catalog) consecutive faults, every type appears (shuffled rounds)."""
    plans = plan_faults(CFG["faults"], 48 * 3600, seed=3)
    k = len(CFG["faults"]["catalog"])
    for i in range(0, len(plans) - k + 1, k):
        assert len({p.type for p in plans[i:i + k]}) == k
    # severities alternate within a type
    for t in {p.type for p in plans}:
        sev = [p.severity for p in plans if p.type == t]
        assert all(x != y for x, y in zip(sev, sev[1:]))


def test_traffic_stream_is_reproducible():
    it1, it2 = iter_traffic(CFG["traffic"], 5), iter_traffic(CFG["traffic"], 5)
    assert [next(it1) for _ in range(50)] == [next(it2) for _ in range(50)]


def _run(plans, injectors, tmp_path, end_s=10 * 3600):
    clock = SimClock(1_800_000_000_000)
    lw = LabelWriter(tmp_path / "labels.csv")
    FaultRunner(plans, injectors, clock, lw, "r", 60, clock.now_ms() + int(end_s * 1000)).run()
    return validate_labels(pd.read_csv(tmp_path / "labels.csv"))


def test_runner_times_and_gaps(tmp_path):
    plans = plan_faults(CFG["faults"], 10 * 3600, seed=2)
    clock_inj = None
    labels = _run(plans, {"mininet": SimInjector(SimClock(0)), "chaos-mesh": SimInjector(SimClock(0)),
                          "traffic": SimInjector(SimClock(0))}, tmp_path)
    # SimInjector above has its own clock -> times would not advance on the runner clock; use shared
    clock = SimClock(1_800_000_000_000)
    inj = SimInjector(clock)
    lw = LabelWriter(tmp_path / "l2.csv")
    FaultRunner(plans, {"mininet": inj, "chaos-mesh": inj, "traffic": inj}, clock, lw, "r", 60,
                clock.now_ms() + 10 * 3600 * 1000).run()
    lab = pd.read_csv(tmp_path / "l2.csv").sort_values("t_applied")
    assert (lab["status"] == "ok").all()
    # the gap rule is measured from the end of the previous EFFECT window
    gaps = lab["t_applied"].iloc[1:].to_numpy() - (lab["t_removed"].iloc[:-1].to_numpy() + 60_000)
    assert (gaps >= 15 * 60_000).all()
    params = json.loads(lab.iloc[0]["params"])
    assert params["apply_latency_ms"] > 0 and "t_requested" in params
    # pod_kill: t_removed = replacement Ready, i.e. includes the recovery time
    pk = lab[lab["type"] == "pod_kill"]
    if len(pk):
        assert ((pk["t_removed"] - pk["t_applied"]) >= SimInjector.RECOVER_MS["pod_kill"]).all()


def test_failed_injection_is_labelled_failed(tmp_path):
    class Boom(SimInjector):
        def apply(self, plan, run_id):
            raise InjectionError("no")
    clock = SimClock(1_800_000_000_000)
    inj = Boom(clock)
    plans = [FaultPlan("f000", "cpu", "mild", {"workers": 1, "load": 50}, "pod:upf", "chaos-mesh", 60, 120)]
    FaultRunner(plans, {"chaos-mesh": inj}, clock, LabelWriter(tmp_path / "l.csv"), "r", 60,
                clock.now_ms() + 3600_000).run()
    lab = pd.read_csv(tmp_path / "l.csv")
    assert lab["status"].tolist() == ["failed"]


def test_chaos_manifests():
    p = FaultPlan("f003", "cpu", "strong", {"workers": 2, "load": 100}, "pod:upf", "chaos-mesh", 0, 300)
    m = chaos_manifest(p, "2026-12-01_120000_run", "5g")
    assert m["kind"] == "StressChaos" and m["spec"]["stressors"]["cpu"] == {"workers": 2, "load": 100}
    assert m["spec"]["duration"] == "420s"                 # planned + safety margin
    assert m["spec"]["selector"]["labelSelectors"] == {"app": "upf"}
    assert len(m["metadata"]["name"]) <= 63 and m["metadata"]["name"] == m["metadata"]["name"].lower()
    mem = chaos_manifest(FaultPlan("f004", "mem", "mild", {"size": "256MB"}, "pod:upf", "chaos-mesh", 0, 60), "r", "5g")
    assert mem["spec"]["stressors"]["memory"]["size"] == "256MB"
    pk = chaos_manifest(FaultPlan("f005", "pod_kill", "mild", {}, "pod:amf", "chaos-mesh", 0, 60), "r", "5g")
    assert pk["kind"] == "PodChaos" and pk["spec"]["action"] == "pod-kill" and pk["spec"]["mode"] == "one"
    with pytest.raises(InjectionError):
        chaos_manifest(FaultPlan("x", "delay", "mild", {}, "pod:upf", "chaos-mesh", 0, 1), "r", "5g")


def test_mininet_injector_over_socket(tmp_path):
    """The injector speaks the mininet_service.py protocol; a fake server records the requests."""
    path = str(tmp_path / "mn.sock")
    got = []
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(4)

    def serve():
        for _ in range(2):
            c, _ = srv.accept()
            req = json.loads(c.recv(65536).decode())
            got.append(req)
            c.sendall((json.dumps({"ok": True, "t_ms": 1234}) + "\n").encode())
            c.close()
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    inj = MininetInjector(SimClock(0), MininetRPC(path))
    p = FaultPlan("f1", "delay", "mild", {"delay_ms": 10, "jitter_ms": 2}, "link:s1-s2", "mininet", 0, 60)
    assert inj.apply(p, "r") == 1234
    assert inj.remove(p, "r") == 1234
    th.join(2)
    assert got[0] == {"op": "impair", "a": "s1", "b": "s2", "delay_ms": 10, "jitter_ms": 2, "loss_pct": 0}
    assert got[1] == {"op": "reset", "a": "s1", "b": "s2"}
