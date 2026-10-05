#!/usr/bin/env python3
"""End-to-end check of the SDN transport, without Kubernetes. Run as root.

Starts Ryu (ring_router.py) and the Mininet service, then plugs two network namespaces into the
ring exactly the way OVS-CNI plugs the gNB and UPF pods in (a veth port on bridge s1 / s3,
10.100.3.10 and 10.100.3.20). Checks, in order:
  1. the controller's forwarding tree is installed and the namespaces reach each other
  2. traffic follows s1-s2-s3; s3-s4 carries none (it is outside the forwarding tree)
  3. delay injection on s2-s3 raises the RTT by ~2x the injected one-way delay
  4. link_down on s1-s2 interrupts traffic, which comes back over s1-s4-s3 after reconvergence;
     the outage duration is measured
  5. restoring the link brings traffic back to s1-s2-s3
Results go to infra/sdn/check_ring_result.json.

  RYU_MANAGER=/path/to/ryu-manager  DATAPATH=user|kernel  sudo -E python3 infra/sdn/check_ring.py
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RYU = os.environ.get("RYU_MANAGER", "ryu-manager")
DATAPATH = os.environ.get("DATAPATH", "kernel")
SOCK = "/run/testbed-mininet-check.sock"
results = []
procs = []


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr}")
    return r.stdout


def rpc(**req):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(20)
        s.connect(SOCK)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            c = s.recv(65536)
            if not c:
                break
            buf += c
    return json.loads(buf)


def scrape() -> dict:
    txt = urllib.request.urlopen("http://127.0.0.1:9300/metrics", timeout=3).read().decode()
    out = {}
    for m in re.finditer(r'^ryu_port_tx_bytes_total\{.*?link="([^"]+)".*?\} ([0-9.e+]+)$', txt, re.M):
        out[m.group(1)] = float(m.group(2))
    fwd = {m.group(1): float(m.group(2)) for m in
           re.finditer(r'^ryu_link_forwarding\{link="([^"]+)"\} ([0-9.]+)$', txt, re.M)}
    return {"tx": out, "fwd": fwd}


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def ping(n=10, interval=0.2):
    out = sh(f"ip netns exec ran ping -n -c {n} -i {interval} -W 1 10.100.3.20")
    rtts = [float(x) for x in re.findall(r"time=([\d.]+) ms", out)]
    return rtts


def traffic_delta(seconds=6):
    a = scrape()["tx"]
    sh(f"ip netns exec ran ping -n -c {int(seconds / 0.01)} -i 0.01 -s 1200 -W 1 10.100.3.20")
    time.sleep(2.5)   # > one Ryu stats poll
    b = scrape()["tx"]
    return {k: b.get(k, 0) - a.get(k, 0) for k in b}


def wait_connectivity(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ping(1, 0.2):
            return time.time() - t0
        time.sleep(0.2)
    return None


def cleanup():
    for p in reversed(procs):
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=20)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    for ns in ("ran", "core"):
        sh(f"ip netns del {ns}")
    sh("mn -c")


def attach(ns, bridge, ip):
    """What OVS-CNI does for a pod: veth pair, one end in the pod netns, one end on the bridge."""
    sh(f"ip netns add {ns}", check=True)
    sh(f"ip link add v-{ns} type veth peer name n3 netns {ns}", check=True)
    sh(f"ovs-vsctl add-port {bridge} v-{ns}", check=True)
    sh(f"ip link set v-{ns} up", check=True)
    sh(f"ip netns exec {ns} ip addr add {ip}/24 dev n3", check=True)
    sh(f"ip netns exec {ns} ip link set n3 up; ip netns exec {ns} ip link set lo up", check=True)


def netem_available() -> bool:
    sh("ip link add tbchk0 type veth peer name tbchk1")
    ok = subprocess.run("tc qdisc add dev tbchk0 root netem delay 1ms", shell=True,
                        capture_output=True).returncode == 0
    sh("ip link del tbchk0")
    return ok


def main():
    netem = netem_available()
    if not netem:
        print("NOTE: kernel has no sch_netem -> ring runs with --no-tc; the delay check is SKIPPED.")
    sh("mn -c")
    procs.append(subprocess.Popen([RYU, str(HERE / "ryu_apps/ring_router.py")], stdout=open(HERE / "ryu_check.log", "w"),
                                  stderr=subprocess.STDOUT, preexec_fn=os.setsid))
    time.sleep(3)
    procs.append(subprocess.Popen([sys.executable, "-u", str(HERE / "mininet_service.py"), "--datapath", DATAPATH,
                                   "--socket", SOCK] + ([] if netem else ["--no-tc"]), stdout=open(HERE / "mininet_check.log", "w"),
                                  stderr=subprocess.STDOUT, preexec_fn=os.setsid))
    for _ in range(240):                     # the userspace datapath can take a minute to come up
        if os.path.exists(SOCK):
            break
        time.sleep(0.5)
    time.sleep(2)
    try:
        st = rpc(op="status")
        check("Mininet ring up, 4 switches connected to Ryu", all(s["connected"] for s in st["switches"])
              and len(st["switches"]) == 4, str(st["switches"]))
        attach("ran", "s1", "10.100.3.10")
        attach("core", "s3", "10.100.3.20")
        dt = wait_connectivity(120)
        check("controller tree up, RAN site reaches core site", dt is not None, f"after {dt:.1f} s" if dt else "")
        time.sleep(3)
        f = scrape()["fwd"]
        check("tree excludes s3-s4 only", f.get("s3-s4") == 0 and all(f.get(l) == 1 for l in ("s1-s2", "s2-s3", "s4-s1")),
              json.dumps(f))
        d = traffic_delta()
        on_path = d.get("s1-s2", 0) > 1e5 and d.get("s2-s3", 0) > 1e5 and d.get("s3-s4", 0) < 1e4
        check("traffic follows s1-s2-s3", on_path, json.dumps({k: int(v) for k, v in d.items()}))
        import statistics as stt
        base = ping(20)
        imp = base
        if netem:
            rpc(op="impair", a="s2", b="s3", delay_ms=20, jitter_ms=0, loss_pct=0)
            time.sleep(1)
            imp = ping(20)
            rpc(op="reset", a="s2", b="s3")
            inc = stt.median(imp) - stt.median(base)
            check("delay 20 ms on s2-s3 adds ~40 ms RTT", 35 <= inc <= 48,
                  f"median RTT {stt.median(base):.2f} -> {stt.median(imp):.2f} ms (+{inc:.1f})")
        else:
            results.append({"check": "delay 20 ms on s2-s3 adds ~40 ms RTT", "ok": None,
                            "detail": "skipped: no sch_netem"})
            print("[SKIP] delay injection (no sch_netem in this kernel)")
        t_down = rpc(op="link", a="s1", b="s2", state="down")["t_ms"]
        out_s = wait_connectivity(180)
        check("link_down s1-s2: traffic restored via s4", out_s is not None,
              f"outage {out_s:.2f} s (controller reroute)" if out_s is not None else "never restored")
        time.sleep(3)
        d2 = traffic_delta()
        check("rerouted path is s1-s4-s3", d2.get("s4-s1", 0) > 1e5 and d2.get("s3-s4", 0) > 1e5 and d2.get("s1-s2", 0) < 1e4,
              json.dumps({k: int(v) for k, v in d2.items()}))
        rpc(op="link", a="s1", b="s2", state="up")
        back = wait_connectivity(180)
        time.sleep(3)
        d3 = traffic_delta()
        check("after link up, traffic back on s1-s2-s3", d3.get("s1-s2", 0) > 1e5 and d3.get("s3-s4", 0) < 1e4,
              json.dumps({k: int(v) for k, v in d3.items()}))
        results.append({"measured": {"initial_connectivity_s": dt, "link_up_outage_s": back, "link_down_outage_s": out_s,
                                     "rtt_base_ms": stt.median(base), "rtt_delay20_ms": stt.median(imp)}})
    finally:
        cleanup()
        (HERE / "check_ring_result.json").write_text(json.dumps(
            {"datapath": DATAPATH, "netem_available": netem, "results": results}, indent=1))
    n_fail = sum(1 for r in results if r.get("ok") is False)
    results_meta = {"netem": netem}
    print(f"\n{sum(1 for r in results if r.get('ok'))} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
