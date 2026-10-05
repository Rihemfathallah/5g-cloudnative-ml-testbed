#!/usr/bin/env python3
"""UE-side probe: what the user experiences, as Prometheus metrics on :9200.

Runs in the UE pod, sharing srsUE's network namespace, so it sees the tunnel interface.
  probe_rtt_ms              last ping RTT through the 5G tunnel (NaN when no reply)
  probe_loss_ratio          share of the last 10 pings without reply
  probe_tun_rx_bytes_total  bytes received on the tunnel (counter; resets when srsUE re-attaches)
  probe_tun_tx_bytes_total  bytes sent on the tunnel
  probe_registered          1 if the tunnel exists and has an IPv4 address

One ping per second with a 1 s timeout keeps the probe's own load (~100 B/s) negligible.
`idle` traffic episodes write a pause (seconds) to /run/probe/pause_s to get truly quiet periods.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path

from prometheus_client import Gauge, start_http_server
from prometheus_client.core import CounterMetricFamily, REGISTRY

UE = os.environ.get("UE_NAME", "ue1")
DEV = os.environ.get("TUN_DEV", "tun_srsue")
TARGET = os.environ.get("PING_TARGET", "10.45.0.1")
PAUSE = Path("/run/probe/pause_s")
_RTT_RE = re.compile(r"time=([\d.]+) ms")


def tun_ipv4(dev: str) -> str | None:
    r = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", dev], capture_output=True, text=True)
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", r.stdout)
    return m.group(1) if m else None


def ping_once(dev: str, target: str) -> float | None:
    r = subprocess.run(["ping", "-n", "-c", "1", "-W", "1", "-I", dev, target],
                       capture_output=True, text=True)
    m = _RTT_RE.search(r.stdout)
    return float(m.group(1)) if m else None


def read_counter(dev: str, name: str) -> float:
    try:
        return float(Path(f"/sys/class/net/{dev}/statistics/{name}").read_text())
    except OSError:
        return math.nan


class TunCounters:
    def collect(self):
        for direction, f in (("rx", "rx_bytes"), ("tx", "tx_bytes")):
            c = CounterMetricFamily(f"probe_tun_{direction}_bytes", f"tunnel {direction} bytes", labels=["ue"])
            v = read_counter(DEV, f)
            if not math.isnan(v):
                c.add_metric([UE], v)
            yield c


def main() -> None:
    rtt = Gauge("probe_rtt_ms", "ping RTT through the tunnel", ["ue"])
    loss = Gauge("probe_loss_ratio", "share of the last 10 pings lost", ["ue"])
    reg = Gauge("probe_registered", "tunnel up with IPv4", ["ue"])
    REGISTRY.register(TunCounters())
    start_http_server(int(os.environ.get("PORT", "9200")))
    window: deque[bool] = deque(maxlen=10)
    pause_until = 0.0
    while True:
        t0 = time.time()
        if PAUSE.exists():
            try:
                pause_until = t0 + float(PAUSE.read_text().strip())
            finally:
                PAUSE.unlink(missing_ok=True)
        ip = tun_ipv4(DEV)
        reg.labels(UE).set(1.0 if ip else 0.0)
        if t0 >= pause_until:
            r = ping_once(DEV, TARGET) if ip else None
            window.append(r is not None)
            rtt.labels(UE).set(r if r is not None else math.nan)
            loss.labels(UE).set(1.0 - sum(window) / len(window))
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))


if __name__ == "__main__":
    main()
