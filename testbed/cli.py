"""`tb` command line. Every experiment in the report is one of these commands.

  tb plan     configs/runs/X.yaml                 print the fault plan, inject nothing
  tb run      configs/runs/X.yaml [--force]       execute a run (testbed or synthetic)
  tb export   runs/<id> --start .. --end ..       re-export a run from Prometheus
  tb windows  runs/<id> [runs/<id> ...] -o data/<name> [--test-only runs/<id>]
  tb detect   data/<name> -o results/<name> [--models ...] [--seeds ...]
  tb smoke                                          synthetic end-to-end check (results/smoke)
  tb handover traces|export-trajectories|tune|all -o results/handover
  tb bgp      download|features|detect
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from testbed.common.config import REPO_ROOT, load_yaml, resolve


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def cmd_plan(a):
    from testbed.control.orchestrate import plan_only
    plans = plan_only(a.config)
    t = 0.0
    for p in plans:
        t += p["gap_before_s"]
        print(f"{p['fault_id']}  +{t / 60:7.1f} min  {p['type']:9s} {p['severity']:6s} "
              f"{p['target']:12s} {p['duration_s'] / 60:5.1f} min  {json.dumps(p['params'])}")
        t += p["duration_s"]
    from collections import Counter
    print("per type/severity:", dict(Counter((p["type"], p["severity"]) for p in plans)))


def cmd_run(a):
    from testbed.control.orchestrate import execute_run
    d = execute_run(a.config, a.runs_root, force=a.force)
    print(d)


def cmd_export(a):
    from testbed.common.config import parse_duration
    from testbed.data.exporter import export_run
    man = load_yaml(Path(a.run_dir) / "manifest.yaml")
    cfg = man["config"]
    coll = man.get("collection", {})
    export_run(cfg["monitoring"]["prometheus_url"], resolve(cfg["monitoring"]["metric_map"]),
               man["run_id"], cfg["run"].get("source", "5g"),
               int(a.start or coll["start_ms"]), int(a.end or coll["end_ms"]), Path(a.run_dir),
               parse_duration(cfg["monitoring"].get("export_chunk", "6h")))


def cmd_windows(a):
    from testbed.data.features import build_windows
    det_cfg = load_yaml(a.config)
    if a.exclude:
        det_cfg.setdefault("columns", {})["exclude"] = list(a.exclude)
    runs = [Path(r) for r in a.runs] + [Path(r) for r in a.test_only]
    ws = build_windows(runs, det_cfg, a.profile, test_only={Path(r).name for r in a.test_only})
    ws.save(Path(a.out))
    (Path(a.out) / "runs.json").write_text(json.dumps([str(r) for r in runs], indent=1))
    print(json.dumps({k: ws.report[k] for k in ("n_windows", "label_counts", "split_counts",
                                                  "columns_kept", "dropped_columns")}, indent=1))


def cmd_detect(a):
    from testbed.anomaly.experiment import load_faults, run_detection
    from testbed.data.features import WindowSet
    det_cfg = load_yaml(a.config)
    ws = WindowSet.load(Path(a.data))
    runs = [Path(r) for r in json.loads((Path(a.data) / "runs.json").read_text())]
    summary = run_detection(ws, load_faults(runs), det_cfg, Path(a.out), a.models,
                            [int(s) for s in a.seeds] if a.seeds else None)
    print(summary.to_string(index=False))


def cmd_smoke(a):
    """Synthetic end to end. Proves the plumbing, says nothing about 5G."""
    from testbed.anomaly.experiment import load_faults, run_detection
    from testbed.control.orchestrate import execute_run
    from testbed.data.features import build_windows
    root = REPO_ROOT / "runs"
    d = execute_run("configs/runs/smoke_synthetic.yaml", root)
    det_cfg = load_yaml("configs/detection.yaml")
    if a.fast:
        det_cfg["models"]["lstm_ae"]["epochs"] = 5
        det_cfg["evaluation"]["seeds"] = [0, 1]
    ws = build_windows([d], det_cfg, "testbed")
    out = REPO_ROOT / "results" / "smoke"
    ws.save(out / "data")
    (out / "data" / "runs.json").write_text(json.dumps([str(d)]))
    s = run_detection(ws, load_faults([d]), det_cfg, out, tag={"synthetic": True})
    (out / "README_SYNTHETIC.txt").write_text(
        "These results come from testbed/sim/synthetic_testbed.py, a statistical stand-in used to\n"
        "test the pipeline. They are NOT results about Open5GS, srsRAN or any 5G network and must\n"
        "never be reported, plotted in the report, or put on a CV.\n")
    print(s.to_string(index=False))


def cmd_handover(a):
    from testbed.handover import pipeline
    pipeline.main(a.stage, load_yaml(a.config), Path(a.out), fast=a.fast)


def cmd_bgp(a):
    from testbed.bgp import pipeline as bp
    bp.main(a.stage, load_yaml(a.config), load_yaml(a.det_config), Path(a.out),
            allow_unverified=a.allow_unverified)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tb")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("plan"); s.add_argument("config"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("run"); s.add_argument("config"); s.add_argument("--force", action="store_true")
    s.add_argument("--runs-root", default=None); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("export"); s.add_argument("run_dir"); s.add_argument("--start"); s.add_argument("--end")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("windows"); s.add_argument("runs", nargs="+"); s.add_argument("-o", "--out", required=True)
    s.add_argument("--test-only", nargs="*", default=[]); s.add_argument("--profile", default="testbed")
    s.add_argument("--exclude", nargs="*", default=[], help="regexes of columns to drop (ablations)")
    s.add_argument("--config", default="configs/detection.yaml"); s.set_defaults(fn=cmd_windows)
    s = sub.add_parser("detect"); s.add_argument("data"); s.add_argument("-o", "--out", required=True)
    s.add_argument("--models", nargs="*"); s.add_argument("--seeds", nargs="*")
    s.add_argument("--config", default="configs/detection.yaml"); s.set_defaults(fn=cmd_detect)
    s = sub.add_parser("smoke"); s.add_argument("--fast", action="store_true"); s.set_defaults(fn=cmd_smoke)
    s = sub.add_parser("handover"); s.add_argument("stage", choices=["traces", "export-trajectories", "tune", "all"])
    s.add_argument("-o", "--out", default="results/handover"); s.add_argument("--config", default="configs/handover.yaml")
    s.add_argument("--fast", action="store_true", help="few trajectories, for CI only"); s.set_defaults(fn=cmd_handover)
    s = sub.add_parser("bgp"); s.add_argument("stage", choices=["download", "features", "detect", "all"])
    s.add_argument("-o", "--out", default="results/bgp"); s.add_argument("--config", default="configs/bgp.yaml")
    s.add_argument("--det-config", default="configs/detection.yaml")
    s.add_argument("--allow-unverified", action="store_true"); s.set_defaults(fn=cmd_bgp)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
