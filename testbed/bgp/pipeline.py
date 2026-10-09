"""RQ1b: BGP events as runs of the same data contract, then the SAME detectors.

Each event becomes runs/bgp_<event>/ with raw.parquet (per-minute features) and labels.csv
(one row per anomalous interval, type bgp_event). Protocol: per-event time split (the default
pipeline) AND leave-one-event-out (train on two events' training portions, test on the third).
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from testbed.common.config import REPO_ROOT
from testbed.common.schema import LABEL_COLUMNS

log = logging.getLogger(__name__)


def _ms(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def event_labels(event: dict) -> pd.DataFrame:
    rows = [{"fault_id": f"{event['name']}{i}", "type": "bgp_event", "severity": "event",
             "params": "{}", "target": "internet", "injector": "none",
             "t_applied": _ms(a), "t_removed": _ms(b), "status": "ok"}
            for i, (a, b) in enumerate(event["anomalous"])]
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def main(stage: str, cfg: dict, det_cfg: dict, out: Path, allow_unverified: bool = False) -> None:
    from testbed.anomaly.experiment import load_faults, run_detection
    from testbed.bgp.download import download_event
    from testbed.bgp.features import features_from_files, to_canonical
    from testbed.data.features import build_windows

    data_dir = REPO_ROOT / "bgp_data"
    run_dirs = []
    for ev in cfg["events"]:
        if not ev.get("verified") and not allow_unverified:
            raise SystemExit(f"event {ev['name']}: anomalous interval not verified against a paper "
                             "(set verified: true after checking, or --allow-unverified for tests)")
        rd = REPO_ROOT / "runs" / f"bgp_{ev['name']}"
        rd.mkdir(parents=True, exist_ok=True)
        if stage in ("download", "all"):
            man = download_event(ev, cfg["collectors"], data_dir)
            man.to_csv(rd / "download_manifest.csv", index=False)
        if stage in ("features", "all"):
            files = sorted((data_dir / ev["name"]).rglob("updates.*"))
            if not files:
                raise SystemExit(f"no MRT files for {ev['name']}; run `tb bgp download` first")
            feats = features_from_files(files, cfg.get("peer_filter"))
            to_canonical(feats, f"bgp_{ev['name']}", "bgp").to_parquet(rd / "raw.parquet", index=False)
            event_labels(ev).to_csv(rd / "labels.csv", index=False)
        run_dirs.append(rd)
    if stage not in ("detect", "all"):
        return
    # per-event time split
    for rd in run_dirs:
        ws = build_windows([rd], det_cfg, "bgp")
        run_detection(ws, load_faults([rd]), det_cfg, out / rd.name, tag={"protocol": "time_split"})
    # leave one event out
    if cfg.get("protocol") == "leave_one_event_out" and len(run_dirs) >= 2:
        for test_rd in run_dirs:
            train = [r for r in run_dirs if r != test_rd]
            ws = build_windows(train + [test_rd], det_cfg, "bgp", test_only={test_rd.name})
            run_detection(ws, load_faults(train + [test_rd]), det_cfg, out / f"loeo_{test_rd.name}",
                          tag={"protocol": "leave_one_event_out", "test_event": test_rd.name})
