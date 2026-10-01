"""SYNTHETIC STAND-IN for the 5G testbed. Test infrastructure only.

It turns a run's context.jsonl (traffic actually played) and labels.csv (faults actually applied,
from the real scheduler running on a simulated clock) into a canonical raw table with plausible
dynamics, so that every downstream component can be exercised end to end in CI.

What it is NOT: a model of Open5GS. Effect sizes are hand-picked. Any number computed on this data
is a property of this file, not of a 5G network. Results go to results/smoke/ (git-ignored).

Deliberate realism, so that the pipeline is tested against the things that break detectors:
* traffic-driven variability in normal operation (bursts, idle, UE re-attach outages)
* nuisance events that are NOT labelled (MongoDB compaction spikes, scrape gaps)
* mild faults whose effect is close to the noise floor
* recovery tails that outlast t_removed (path switch-back, NF re-registration)
* missing data while a killed pod is being replaced
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from testbed.common.schema import validate_raw

STEP_MS = 5000
NFS = ["amf", "smf", "upf", "nrf", "ausf", "udm", "udr", "pcf", "nssf", "bsf", "mongodb", "gnb", "ue"]
RING = ["s1-s2", "s2-s3", "s3-s4", "s4-s1"]
ACTIVE_PATH = ["s1-s2", "s2-s3"]      # controller tree rooted at s1; gNB on s1, UPF on s3
BACKUP_PATH = ["s3-s4", "s4-s1"]
REROUTE_S = 1.0                       # flow flush + re-learn; measured ~0.01 s by infra/sdn/check_ring.py,
                                      # 1 s leaves room for TCP recovery after the flush


def _ar1(rng, n, phi, sigma):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(max(1e-9, 1 - phi**2))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _mask(t, a_ms, b_ms):
    return (t >= a_ms) & (t < b_ms)


def generate(run_dir: Path, run_id: str, start_ms: int, end_ms: int, seed: int,
             source: str = "synthetic") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(start_ms - start_ms % STEP_MS, end_ms, STEP_MS, dtype=np.int64)
    n = len(t)

    # ------------------------------------------------------------------ traffic from context
    ul_mbps, dl_mbps = np.zeros(n), np.zeros(n)
    detached = np.zeros(n, bool)
    idle = np.zeros(n, bool)
    ctx_path = run_dir / "context.jsonl"
    if ctx_path.exists():
        for line in ctx_path.read_text().splitlines():
            ep = json.loads(line)
            m = _mask(t, ep["t_start"], ep["t_end"])
            if ep["kind"] == "iperf_burst":
                (dl_mbps if ep.get("reverse") else ul_mbps)[m] += ep["mbps"]
            elif ep["kind"] == "reattach":
                detached |= _mask(t, ep["t_start"], ep["t_end"] - 4000)
            elif ep["kind"] == "idle":
                idle |= m
    # throughput is capped by the emulated radio (~ 40 Mb/s DL / 15 Mb/s UL in ZMQ setups)
    ul_mbps, dl_mbps = np.minimum(ul_mbps, 15.0), np.minimum(dl_mbps, 40.0)

    # ------------------------------------------------------------------ baseline signals
    rtt = 9.0 + 0.08 * (ul_mbps + dl_mbps) + np.exp(_ar1(rng, n, 0.9, 0.08)) * 2.0
    loss = np.clip(0.002 + np.abs(_ar1(rng, n, 0.7, 0.002)), 0, 1)
    registered = (~detached).astype(float)
    upf_cpu = 0.02 + 0.004 * (ul_mbps + dl_mbps) + np.abs(_ar1(rng, n, 0.8, 0.004))
    gnb_cpu = 0.85 + 0.01 * (ul_mbps + dl_mbps) + _ar1(rng, n, 0.9, 0.02)  # ZMQ busy-loop
    ue_cpu = 0.70 + 0.008 * (ul_mbps + dl_mbps) + _ar1(rng, n, 0.9, 0.02)
    cpu = {nf: 0.004 + np.abs(_ar1(rng, n, 0.8, 0.002)) for nf in NFS}
    cpu.update(upf=upf_cpu, gnb=gnb_cpu, ue=ue_cpu)
    mem = {nf: (40e6 + 20e6 * rng.random()) * (1 + 0.0000005 * np.arange(n)) + _ar1(rng, n, 0.95, 2e5)
           for nf in NFS}
    mem["mongodb"] += 150e6
    mem["gnb"] += 600e6
    mem["ue"] += 300e6
    traffic_Bps = (ul_mbps + dl_mbps) * 1e6 / 8 + 120.0 * (~idle)   # + 1 Hz ping
    n3 = {"upf": traffic_Bps * 1.06, "gnb": traffic_Bps * 1.06}      # GTP-U overhead
    path_load = {l: np.zeros(n) for l in RING}
    on_backup = np.zeros(n, bool)
    gnb_conn = np.ones(n)
    ran_ue = registered.copy()
    reg_rate = np.zeros(n)
    nan_mask = {nf: np.zeros(n, bool) for nf in NFS}

    # registrations after each re-attach
    starts = np.flatnonzero(np.diff(detached.astype(int)) == -1) + 1
    for s in starts:
        reg_rate[s:s + 6] += 1 / 30.0
        cpu["amf"][s:s + 3] += 0.02
        cpu["smf"][s:s + 3] += 0.015
        cpu["udm"][s:s + 2] += 0.01

    # ------------------------------------------------------------------ unlabeled nuisance
    t_h = (end_ms - start_ms) / 3.6e6
    for _ in range(rng.poisson(t_h / 2.0)):        # MongoDB compaction / WiredTiger checkpoint
        i = int(rng.integers(n))
        w = int(rng.integers(2, 8))
        cpu["mongodb"][i:i + w] += rng.uniform(0.05, 0.2)
        cpu["udr"][i:i + w] += rng.uniform(0.005, 0.02)
    for _ in range(rng.poisson(t_h / 3.0)):        # a slow scrape -> transient RTT outlier
        i = int(rng.integers(n))
        rtt[i] += rng.uniform(5, 20)

    # ------------------------------------------------------------------ faults
    labels_path = run_dir / "labels.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    for _, f in labels.iterrows():
        if f["status"] != "ok":
            continue
        a, b = int(f["t_applied"]), int(f["t_removed"])
        p = json.loads(f["params"])
        m = _mask(t, a, b)
        link = f["target"][5:] if str(f["target"]).startswith("link:") else None
        if f["type"] == "delay":
            d = 2 * p["delay_ms"] if link in ACTIVE_PATH else 0.0   # applied in both directions
            rtt[m] += d + rng.normal(0, p.get("jitter_ms", 0) * 1.4, m.sum())
            ul_mbps[m] *= 1 / (1 + p["delay_ms"] / 40)              # TCP slows with RTT
            dl_mbps[m] *= 1 / (1 + p["delay_ms"] / 40)
        elif f["type"] == "loss":
            if link in ACTIVE_PATH:
                lp = 1 - (1 - p["loss_pct"] / 100) ** 2
                loss[m] += lp + rng.normal(0, lp / 3, m.sum()).clip(-lp, None)
                ul_mbps[m] *= max(0.15, 1 - 8 * lp)
                dl_mbps[m] *= max(0.15, 1 - 8 * lp)
        elif f["type"] == "link_down":
            if link not in ACTIVE_PATH:
                continue
            rec = int(REROUTE_S * 1000)
            for o in (_mask(t, a, a + rec), _mask(t, b, b + rec)):   # flush + re-learn, both ways
                loss[o] += 0.2
            on_backup |= _mask(t, a, b)
            # path s1-s4-s3 has the same hop count as s1-s2-s3: no RTT change by design
        elif f["type"] == "cpu":
            load = p["workers"] * p["load"] / 100
            cpu["upf"][m] += np.minimum(load, 2.0) * rng.uniform(0.9, 1.0, m.sum())
            rtt[m] += (2.0 + 6.0 * (load - 0.5)) * np.abs(rng.normal(1, 0.3, m.sum()))
        elif f["type"] == "mem":
            size = float(str(p["size"]).upper().replace("MB", "")) * 1024**2
            ramp = np.clip((t[m] - a) / 20000, 0, 1)   # stress-ng allocates over ~20 s
            mem["upf"][m] += size * ramp
            cpu["upf"][m] += 0.25 * p.get("workers", 1)
        elif f["type"] == "pod_kill":
            app = f["target"][4:]
            gone = _mask(t, a, b)                        # pod absent until replacement Ready
            nan_mask[app] |= gone
            if app == "amf":
                gnb_conn[_mask(t, a, b + 8000)] = 0
                ran_ue[_mask(t, a, b + 15000)] = 0
                reg_rate[_mask(t, b + 8000, b + 40000)] += 1 / 30
            elif app == "smf":
                registered[_mask(t, a, b + 20000)] = 0
            restart_spike = _mask(t, b, b + 20000)
            cpu[app][restart_spike] += 0.05
            cpu["nrf"][restart_spike] += 0.01
        elif f["type"] == "flood":
            ul_mbps[m] = 15.0 * rng.uniform(0.92, 1.0, m.sum())
            rtt[m] += 25 + 0.6 * p["mbps"] * np.abs(rng.normal(1, 0.25, m.sum()))
            loss[m] += 0.01 * p["parallel"]
            cpu["upf"][m] += 0.06 + 0.002 * p["mbps"]

    # detached UE: no tunnel, no ping
    rtt[detached] = np.nan
    loss[detached] = 1.0
    ul_mbps[detached] = dl_mbps[detached] = 0.0
    registered[detached] = 0.0
    traffic_Bps = (ul_mbps + dl_mbps) * 1e6 / 8 + 120.0 * (~idle)
    n3 = {"upf": traffic_Bps * 1.06, "gnb": traffic_Bps * 1.06}
    for l in RING:
        if l in ACTIVE_PATH:
            path_load[l] = np.where(on_backup, 0.0, traffic_Bps * 1.06)
        else:
            path_load[l] = np.where(on_backup, traffic_Bps * 1.06, 0.0)
        path_load[l] = path_load[l] + 30 + np.abs(rng.normal(0, 3, n))  # BPDUs + LLDP-like chatter
    link_fwd = {l: np.where(on_backup, 1.0, 0.0) if l == "s3-s4" else np.ones(n) for l in RING}
    link_up = {l: np.ones(n) for l in RING}
    if len(labels):
        for _, f in labels[(labels["type"] == "link_down") & (labels["status"] == "ok")].iterrows():
            l = f["target"][5:]
            m = _mask(t, int(f["t_applied"]), int(f["t_removed"]))
            link_fwd[l][m] = 0
            link_up[l][m] = 0

    # ------------------------------------------------------------------ assemble long table
    cols: dict[tuple[str, str], np.ndarray] = {}
    for nf in NFS:
        c, mm = cpu[nf].copy(), mem[nf].copy()
        c[nan_mask[nf]] = np.nan
        mm[nan_mask[nf]] = np.nan
        cols[(nf, "cpu_seconds_rate")] = np.clip(c, 0, None)
        cols[(nf, "mem_working_set_bytes")] = mm
    for nf in ("upf", "gnb"):
        cols[(nf, "n3_rx_bytes_rate")] = n3[nf] * rng.uniform(0.98, 1.02, n)
        cols[(nf, "n3_tx_bytes_rate")] = n3[nf] * rng.uniform(0.98, 1.02, n)
    cols[("amf", "gnb_connected")] = gnb_conn
    cols[("amf", "ran_ue")] = ran_ue
    cols[("amf", "reg_init_req_rate")] = reg_rate
    cols[("ue1", "rtt_ms")] = rtt
    cols[("ue1", "loss_ratio")] = np.clip(loss, 0, 1)
    cols[("ue1", "dl_bytes_rate")] = dl_mbps * 1e6 / 8 + 60
    cols[("ue1", "ul_bytes_rate")] = ul_mbps * 1e6 / 8 + 60
    cols[("ue1", "registered")] = registered
    for l in RING:
        cols[(l, "link_tx_bytes_rate")] = path_load[l]
        cols[(l, "link_forwarding")] = link_fwd[l]
        cols[(l, "link_up")] = link_up[l]
    cols[("host", "cpu_busy_ratio")] = np.clip(0.35 + 0.1 * cpu["upf"] + _ar1(rng, n, 0.9, 0.01), 0, 1)

    # scrape gaps: ~0.2 % of samples missing at random (Prometheus skipped a scrape)
    frames = []
    for (ent, met), v in cols.items():
        keep = rng.random(n) > 0.002
        frames.append(pd.DataFrame({"ts": t[keep], "value": v[keep], "entity": ent, "metric": met}))
    df = pd.concat(frames, ignore_index=True).assign(run_id=run_id, source=source)
    df = validate_raw(df)
    df.to_parquet(run_dir / "raw.parquet", index=False)
    return df
