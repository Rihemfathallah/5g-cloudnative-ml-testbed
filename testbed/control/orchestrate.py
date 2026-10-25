"""Run lifecycle: plan -> preflight -> warm-up -> collection -> teardown -> export -> freeze.

One command reproduces any run:   tb run configs/runs/<name>.yaml
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import signal
import threading
from pathlib import Path

from testbed.common.clock import RealClock, SimClock
from testbed.common.config import load_yaml, parse_duration, resolve
from testbed.common.runinfo import (derive_seed, freeze_manifest, make_run_id, run_dir,
                                    seed_everything)
from testbed.control.injectors import (ChaosMeshInjector, MininetInjector, MininetRPC,
                                       SimInjector, TrafficFloodInjector)
from testbed.control.kube import Kube
from testbed.control.runner import FaultRunner, LabelWriter, TrafficRunner
from testbed.control.schedule import iter_traffic, plan_faults

log = logging.getLogger(__name__)

# Synthetic runs start at a fixed instant so that their output is byte-for-byte reproducible.
SYNTH_EPOCH_MS = int(dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)


def plan_only(cfg_path: str) -> list[dict]:
    cfg = load_yaml(cfg_path)
    rc = cfg["run"]
    plans = plan_faults(cfg["faults"], parse_duration(rc["duration"]),
                        derive_seed(int(rc["seed"]), "faults"))
    return [p.__dict__ for p in plans]


def execute_run(cfg_path: str, runs_root: str | Path | None = None, force: bool = False,
                run_id: str | None = None) -> Path:
    cfg = load_yaml(cfg_path)
    rc = cfg["run"]
    seed = int(rc["seed"])
    seed_everything(seed)
    duration_s = parse_duration(rc["duration"])
    warmup_s = parse_duration(rc.get("warmup", "0s"))
    plans = plan_faults(cfg["faults"], duration_s, derive_seed(seed, "faults"))
    grace_s = parse_duration(cfg["faults"].get("grace", "60s"))
    profile = rc["profile"]
    run_id = run_id or (f"synthetic_{rc['name']}_s{seed}" if profile == "synthetic"
                        else make_run_id(rc["name"]))
    d = run_dir(run_id, runs_root)
    for stale in ("labels.csv", "context.jsonl"):   # re-running a deterministic id starts clean
        (d / stale).unlink(missing_ok=True)
    with open(d / "plan.json", "w") as f:
        json.dump([p.__dict__ for p in plans], f, indent=2)
    log.info("run %s: %d faults planned over %.1f h", run_id, len(plans), duration_s / 3600)

    labels = LabelWriter(d / "labels.csv")
    if profile == "synthetic":
        start = SYNTH_EPOCH_MS
        coll_start, end = start + int(warmup_s * 1000), start + int((warmup_s + duration_s) * 1000)
        tclock = SimClock(start)
        TrafficRunner(iter_traffic(cfg["traffic"], derive_seed(seed, "traffic")),
                      _sim_traffic_actions(tclock), tclock, d / "context.jsonl", end).run()
        fclock = SimClock(coll_start)
        inj = SimInjector(fclock)
        FaultRunner(plans, {"mininet": inj, "chaos-mesh": inj, "traffic": inj}, fclock, labels,
                    run_id, grace_s, end).run()
        from testbed.sim.synthetic_testbed import generate
        generate(d, run_id, coll_start, end, derive_seed(seed, "synthetic"), rc.get("source", "synthetic"))
    else:
        coll_start, end = _execute_real(cfg, plans, labels, d, run_id, warmup_s, duration_s,
                                        grace_s, force)
    freeze_manifest(cfg, d, run_id, extra={
        "config_path": str(cfg_path),
        "collection": {"start_ms": coll_start, "end_ms": end},
        "synthetic": profile == "synthetic",
    })
    return d


def _sim_traffic_actions(clock):
    from testbed.control.traffic import SimTraffic
    return SimTraffic(clock).actions()


def _execute_real(cfg, plans, labels, d: Path, run_id, warmup_s, duration_s, grace_s, force):
    from testbed.control.preflight import Prom, report, run_preflight
    from testbed.control.traffic import K8sTraffic
    from testbed.data.exporter import export_run

    rc, mon = cfg["run"], cfg["monitoring"]
    clock, kube = RealClock(), Kube(rc.get("namespace", "5g"))
    prom, mn = Prom(mon["prometheus_url"]), MininetRPC()
    if not report(run_preflight(cfg, kube, prom, mn)) and not force:
        raise SystemExit("preflight failed; fix the testbed or pass --force (the run will be marked)")

    tun = rc.get("ue_tun", "tun_srsue")          # uesimtun0 for UERANSIM, oaitun_ue1 for OAI
    injectors = {"mininet": MininetInjector(clock, mn),
                 "chaos-mesh": ChaosMeshInjector(clock, kube),
                 "traffic": TrafficFloodInjector(clock, kube, dev=tun)}
    stop = threading.Event()

    def _handler(signum, frame):
        log.warning("signal %s: stopping, cleaning up impairments", signum)
        stop.set()
    old = {s: signal.signal(s, _handler) for s in (signal.SIGINT, signal.SIGTERM)}

    t0 = clock.now_ms()
    coll_start = t0 + int(warmup_s * 1000)
    end = coll_start + int(duration_s * 1000)
    traffic = TrafficRunner(iter_traffic(cfg["traffic"], derive_seed(int(rc["seed"]), "traffic")),
                            K8sTraffic(kube, clock, dev=tun).actions(), clock, d / "context.jsonl", end, stop)
    faults = FaultRunner(plans, injectors, clock, labels, run_id, grace_s, end, stop)
    tt = threading.Thread(target=traffic.run, name="traffic", daemon=True)
    try:
        tt.start()
        log.info("warm-up %.0f s (discarded)", warmup_s)
        if clock.sleep(warmup_s, stop):
            faults.run()
            clock.sleep((end - clock.now_ms()) / 1000, stop)
    finally:
        stop.set()
        tt.join(timeout=120)
        for name, inj in injectors.items():
            log.info("teardown: %s", name)
            inj.cleanup_all(run_id)
        for s, h in old.items():
            signal.signal(s, h)
    end = min(end, clock.now_ms())
    export_run(mon["prometheus_url"], resolve(mon["metric_map"]), run_id, rc.get("source", "5g"),
               coll_start, end, d, parse_duration(mon.get("export_chunk", "6h")))
    return coll_start, end
