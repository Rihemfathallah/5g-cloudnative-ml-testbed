#!/usr/bin/env python3
"""Validate the Open5GS configuration by RUNNING it, on one host, without Kubernetes.

Starts MongoDB, the ten Open5GS NFs with configs rendered by render.py (profile `local`), a
UERANSIM gNB and UE, then checks, in order:
  1. every NF starts and stays up (config parsed, ports bound)
  2. the NFs register with the NRF
  3. the gNB completes NG Setup with the AMF
  4. the UE registers and gets a PDU session (tunnel interface with an IP in 10.45.0.0/16)
  5. user-plane packets reach the UPF over GTP-U (UPF N3 counter increases)
  6. every metric name marked `verify: true` in configs/metric_map.yaml exists on /metrics
It writes the metric names it found to infra/open5gs/validated_metrics.txt.

Needs root (TUN devices) and:
  OPEN5GS_BIN  (default /opt/open5gs/bin)    UERANSIM_BIN (default /opt/build/UERANSIM/build)
  MONGOD       (default: mongod on PATH)

Note on the user-plane check: on a single host the UE's tunnel address and the UPF's gateway
10.45.0.1 are both local, so `ping 10.45.0.1` would be answered by the kernel without entering
the tunnel. We send to a non-local address instead and verify that the UPF counted the packets.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OPEN5GS_BIN = Path(os.environ.get("OPEN5GS_BIN", "/opt/open5gs/bin"))
UERANSIM_BIN = Path(os.environ.get("UERANSIM_BIN", "/opt/build/UERANSIM/build"))
MONGOD = os.environ.get("MONGOD", shutil.which("mongod") or "mongod")
ORDER = ["nrf", "ausf", "udm", "udr", "pcf", "nssf", "bsf", "smf", "upf", "amf"]
METRICS = {"amf": "127.0.0.5", "smf": "127.0.0.4", "upf": "127.0.0.7", "pcf": "127.0.0.13"}

procs: list[subprocess.Popen] = []
results: list[tuple[str, bool | None, str]] = []


class _SkipRest(Exception):
    pass


def skip(name: str, why: str) -> None:
    results.append((name, None, why))
    print(f"[SKIP] {name} ({why})", flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def start(cmd: list[str], log: Path) -> subprocess.Popen:
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    procs.append(p)
    return p


def wait_for(pred, timeout: float, poll: float = 0.5) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(poll)
    return False


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def metrics_text(ip: str) -> str:
    try:
        return urllib.request.urlopen(f"http://{ip}:9090/metrics", timeout=3).read().decode()
    except Exception:
        return ""


def metric_value(text: str, name: str) -> float:
    vals = [float(m.group(1)) for m in re.finditer(rf"^{name}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M)]
    return sum(vals) if vals else float("nan")


def cleanup() -> None:
    for p in reversed(procs):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    sh("ip link del ogstun 2>/dev/null")


def sctp_available() -> bool:
    import socket
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM, 132).close()
        return True
    except OSError:
        return False


def main() -> int:
    sctp = sctp_available()
    if not sctp:
        print("NOTE: this kernel has no SCTP (socket(...,132) fails). The AMF needs SCTP for NGAP,")
        print("      so AMF / gNB / UE / user-plane checks are SKIPPED here, not failed.")
        print("      Run this script on the testbed host (sudo modprobe sctp) to cover them.\n")
    work = Path(tempfile.mkdtemp(prefix="o5gs-check-"))
    subprocess.run([sys.executable, str(ROOT / "infra/open5gs/render.py"), "local"], check=True,
                   capture_output=True)
    cfg = ROOT / "build/open5gs-local"
    shutil.rmtree(cfg / "log", ignore_errors=True)    # counts below must come from THIS run
    (cfg / "log").mkdir(exist_ok=True)
    try:
        # --- MongoDB + subscribers
        (work / "db").mkdir()
        start([MONGOD, "--dbpath", str(work / "db"), "--bind_ip", "127.0.0.1", "--quiet"], work / "mongod.log")
        ok = wait_for(lambda: "subscribers" in sh(f"{sys.executable} {ROOT}/infra/open5gs/provision.py "
                                                  "mongodb://127.0.0.1/open5gs 1 2>&1"), 30, 1)
        if not check("MongoDB up and subscriber provisioned", ok):
            return 1
        # --- UPF tunnel interface (what the UPF init script does in the cluster)
        sh("ip tuntap add name ogstun mode tun; ip addr replace 10.45.0.1/16 dev ogstun; ip link set ogstun up")
        # --- NFs
        for nf in ORDER:
            start([str(OPEN5GS_BIN / f"open5gs-{nf}d"), "-c", str(cfg / f"{nf}.yaml")], work / f"{nf}.out")
            time.sleep(1.0 if nf == "nrf" else 0.5)
        time.sleep(4)
        for nf, p in zip(ORDER, procs[1:]):
            if nf == "amf" and not sctp:
                skip("amf running with rendered config", "needs SCTP")
                continue
            alive = p.poll() is None
            detail = "" if alive else (work / f"{nf}.out").read_text()[-600:]
            check(f"{nf} running with rendered config", alive, detail)
        nrf_log = (cfg / "log/nrf.log").read_text() if (cfg / "log/nrf.log").exists() else ""
        registered = sorted(set(re.findall(r"\[(\w+)\] NFInstance registered", nrf_log)) |
                            set(m.lower() for m in re.findall(r"NF registered \[.*?\]\[(\w+)\]", nrf_log)))
        check("NFs registered with NRF", len(re.findall(r"NF registered", nrf_log)) >= 7,
              f"({len(re.findall(r'NF registered', nrf_log))} registrations in nrf.log)")
        peers = metric_value(metrics_text(METRICS["smf"]), "pfcp_peers_active")
        check("N4 (PFCP) association SMF<->UPF up", peers >= 1, f"(pfcp_peers_active={peers:.0f})")
        if not sctp:
            for n in ("gNB NG Setup with AMF", "UE registered + PDU session", "GTP-U packets counted by the UPF"):
                skip(n, "needs SCTP")
        # --- UERANSIM gNB + UE
        g = (ROOT / "infra/ueransim/gnb.yaml.tmpl").read_text()
        for k, v in {"__LINK_IP__": "127.0.0.1", "__NGAP_IP__": "127.0.0.1", "__GTP_IP__": "127.0.0.1",
                     "__AMF_IP__": "127.0.0.5"}.items():
            g = g.replace(k, v)
        (work / "gnb.yaml").write_text(g)
        (work / "ue.yaml").write_text((ROOT / "infra/ueransim/ue.yaml.tmpl").read_text()
                                      .replace("__GNB_IP__", "127.0.0.1"))
        if not sctp:
            raise _SkipRest()
        start([str(UERANSIM_BIN / "nr-gnb"), "-c", str(work / "gnb.yaml")], work / "gnb.log")
        check("gNB NG Setup with AMF",
              wait_for(lambda: "NG Setup procedure is successful" in (work / "gnb.log").read_text(), 20))
        start([str(UERANSIM_BIN / "nr-ue"), "-c", str(work / "ue.yaml")], work / "ue.log")
        got_ip = wait_for(lambda: "10.45." in sh("ip -4 addr show dev uesimtun0 2>/dev/null"), 30)
        check("UE registered + PDU session (uesimtun0 has 10.45.x.x)", got_ip,
              sh("ip -4 -o addr show dev uesimtun0 2>/dev/null").strip())
        # --- user plane reaches the UPF
        before = metric_value(metrics_text(METRICS["upf"]), "fivegs_ep_n3_gtp_indatapktn3upf")
        sh("ping -c 20 -i 0.2 -W 1 -I uesimtun0 192.0.2.1 >/dev/null 2>&1")
        time.sleep(1)
        after = metric_value(metrics_text(METRICS["upf"]), "fivegs_ep_n3_gtp_indatapktn3upf")
        check("GTP-U packets counted by the UPF", after - before >= 15, f"(+{after - before:.0f} packets)")
    except _SkipRest:
        pass
    try:
        # --- metric names
        found = {nf: sorted(set(re.findall(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", metrics_text(ip), re.M)))
                 for nf, ip in METRICS.items()}
        out = ROOT / "infra/open5gs/validated_metrics.txt"
        with open(out, "w") as f:
            f.write(f"# Metric names exposed by Open5GS NFs, recorded by local_check.py\n")
            f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S %Z')}  binary: {OPEN5GS_BIN}\n")
            for nf, names in found.items():
                f.write(f"\n[{nf}]\n" + "\n".join(n for n in names if not n.startswith("process_")) + "\n")
        from testbed.data.metric_map import load_metric_map
        queries, _ = load_metric_map(ROOT / "configs/metric_map.yaml")
        all_names = set().union(*found.values())
        for q in queries:
            m = re.search(r"\(([a-z][a-z_0-9]+)\{namespace", q.promql)
            if not m or q.entity not in METRICS or m.group(1).startswith(("container_", "kube_", "node_")):
                continue                    # kubelet / kube-state-metrics, not an Open5GS metric
            name = m.group(1)
            if q.entity == "amf" and not sctp:
                skip(f"metric `{name}` exists ({q.entity}.{q.metric})", "AMF not running without SCTP")
                continue
            check(f"metric `{name}` exists ({q.entity}.{q.metric})", name in all_names)
    finally:
        cleanup()
        print("logs:", work)
    n_fail = sum(1 for _, ok, _ in results if ok is False)
    n_skip = sum(1 for _, ok, _ in results if ok is None)
    (ROOT / "infra/open5gs/local_check_result.json").write_text(json.dumps(
        {"sctp_available": sctp, "open5gs_bin": str(OPEN5GS_BIN),
         "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in results]}, indent=1))
    print(f"\n{len(results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
