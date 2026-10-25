"""Preflight: refuse to start a run on a testbed that is not in a known-good state.

Every check exists because its failure would silently corrupt labels:
* leftover Chaos Mesh resources or netem qdiscs  -> "normal" data that is actually impaired
* UE not registered                             -> the whole run measures an outage
* GTP-U not crossing the OVS ring               -> transport faults have no effect (CV claim false)
"""
from __future__ import annotations

import dataclasses as dc
import logging
import time

import requests

from testbed.control.injectors import MininetRPC
from testbed.control.kube import Kube

log = logging.getLogger(__name__)

PODS = ["nrf", "ausf", "udm", "udr", "pcf", "nssf", "bsf", "amf", "smf", "upf", "mongodb", "gnb", "ue"]


@dc.dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


class Prom:
    def __init__(self, url: str, timeout: float = 10):
        self.url, self.timeout = url.rstrip("/"), timeout

    def instant(self, q: str) -> float | None:
        r = requests.get(f"{self.url}/api/v1/query", params={"query": q}, timeout=self.timeout)
        r.raise_for_status()
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else None


def run_preflight(cfg: dict, kube: Kube, prom: Prom, mn: MininetRPC) -> list[Check]:
    pf = cfg.get("preflight", {})
    out: list[Check] = []

    for app in PODS:
        try:
            out.append(Check(f"pod {app} Ready", kube.pods_ready(f"app={app}")))
        except Exception as e:
            out.append(Check(f"pod {app} Ready", False, str(e)))

    for kind in ("stresschaos", "podchaos", "networkchaos"):
        try:
            items = kube.get_json(kind).get("items", [])
            out.append(Check(f"no leftover {kind}", not items, ", ".join(i["metadata"]["name"] for i in items)))
        except Exception as e:
            out.append(Check(f"no leftover {kind}", False, str(e)))

    try:
        st = mn.call(op="status")
        impaired = [l for l in st["links"] if l["impaired"] or l["state"] != "up"]
        out.append(Check("SDN links up and clean", not impaired, str(impaired)))
        out.append(Check("Ryu controller connected", all(s["connected"] for s in st["switches"])))
    except Exception as e:
        out.append(Check("mininet service reachable", False, str(e)))

    try:
        reg = prom.instant('max(probe_registered{ue="ue1"})')
        out.append(Check("UE registered", reg == 1.0, f"probe_registered={reg}"))
        rtt = prom.instant('max(probe_rtt_ms{ue="ue1"})')
        out.append(Check("tunnel RTT sane", rtt is not None and rtt < pf.get("max_rtt_ms", 50),
                         f"rtt={rtt}"))
    except Exception as e:
        out.append(Check("prometheus reachable", False, str(e)))

    if pf.get("check_n3_on_sdn", True):
        out.append(check_n3_on_sdn(kube, prom, dev=cfg["run"].get("ue_tun", "tun_srsue")))
    return out


def check_n3_on_sdn(kube: Kube, prom: Prom, seconds: int = 15, dev: str = "tun_srsue") -> Check:
    """Push a known amount of traffic through the tunnel and require that the ring carried it.

    Compares bytes on the gNB's n3 interface with bytes summed over the ring links: if GTP-U
    left through eth0 (wrong bind address), n3 stays flat and the check fails.
    """
    q_n3 = 'sum(increase(container_network_transmit_bytes_total{namespace="5g",pod=~"gnb-.*",interface="n3"}[1m]))'
    before = prom.instant(q_n3) or 0.0
    kube.exec("app=ue", "probe", ["iperf3", "-c", "10.45.0.1", "--bind-dev", dev,
                                  "-p", "5201", "-b", "5M", "-t", str(seconds)],
              timeout=seconds + 30, check=False)
    time.sleep(10)  # two scrapes
    after = prom.instant(q_n3) or 0.0
    gained = after - before
    # 5 Mb/s for `seconds` = 0.625 MB/s; require at least half of it on n3
    need = 0.5 * 5e6 / 8 * seconds
    return Check("GTP-U crosses the SDN ring (gNB n3)", gained > need,
                 f"n3 tx gained {gained / 1e6:.2f} MB, need > {need / 1e6:.2f} MB")


def report(checks: list[Check]) -> bool:
    ok = True
    for c in checks:
        log.info("[%s] %s %s", "PASS" if c.ok else "FAIL", c.name, c.detail)
        ok &= c.ok
    return ok
