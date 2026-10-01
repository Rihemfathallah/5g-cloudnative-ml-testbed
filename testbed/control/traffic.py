"""Traffic actions on the real testbed.

The UE pod has two containers: `srsue` (runs infra/docker/srsue/entrypoint.sh, which restarts
srsUE after a configurable pause) and `probe` (iperf3, ping and the metrics exporter), sharing one
network namespace. The UPF pod has a `dn` sidecar running iperf3 servers on 5201 (normal traffic)
and 5202 (flood), reachable at the UE gateway 10.45.0.1 through the 5G tunnel.
"""
from __future__ import annotations

import logging

from testbed.control.kube import Kube
from testbed.control.schedule import TrafficEpisode

log = logging.getLogger(__name__)


class K8sTraffic:
    def __init__(self, kube: Kube, clock, server: str = "10.45.0.1", dev: str = "tun_srsue"):
        self.kube, self.clock, self.server, self.dev = kube, clock, server, dev

    def iperf_burst(self, ep: TrafficEpisode) -> None:
        p = ep.params
        cmd = ["iperf3", "-c", self.server, "--bind-dev", self.dev, "-p", "5201",
               "-b", f"{p['mbps'] / p['parallel']:.2f}M", "-P", str(p["parallel"]),
               "-t", str(max(1, int(round(ep.duration_s)))), "-J"]
        if p.get("reverse"):
            cmd.append("-R")
        # check=False: a burst that fails because of an ongoing fault is normal, not an error
        self.kube.exec("app=ue", "probe", cmd, timeout=ep.duration_s + 60, check=False)

    def wait(self, ep: TrafficEpisode) -> None:
        # The probe pings continuously (1 Hz), so "ping_only" and "idle" differ only in that
        # idle also pauses the probe's ping to give truly quiet periods.
        if ep.kind == "idle":
            self.kube.exec("app=ue", "probe", ["sh", "-c", f"echo {int(ep.duration_s)} > /run/probe/pause_s"],
                           check=False)
        self.clock.sleep(ep.duration_s)

    def reattach(self, ep: TrafficEpisode) -> None:
        self.kube.exec("app=ue", "srsue",
                       ["sh", "-c", f"echo {ep.duration_s:.1f} > /run/srsue/reattach_delay_s && pkill -INT -x srsue"],
                       check=False)
        # down time + registration time (registration is measured by the probe, not assumed)
        self.clock.sleep(ep.duration_s + 10)

    def actions(self) -> dict:
        return {"iperf_burst": self.iperf_burst, "ping_only": self.wait, "idle": self.wait,
                "reattach": self.reattach}


class SimTraffic:
    """Advances a simulated clock; the synthetic generator reads context.jsonl afterwards."""

    def __init__(self, clock):
        self.clock = clock

    def actions(self) -> dict:
        def adv(ep: TrafficEpisode) -> None:
            extra = 10 if ep.kind == "reattach" else 0
            self.clock.sleep(ep.duration_s + extra)
        return {k: adv for k in ("iperf_burst", "ping_only", "idle", "reattach")}
