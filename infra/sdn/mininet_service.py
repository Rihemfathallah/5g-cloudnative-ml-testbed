#!/usr/bin/env python3
"""SDN transport for N3: a ring of four Open vSwitch switches built by Mininet, controlled by Ryu,
plus a small control socket the fault injector uses. Run as root, before the 5G pods start:

    sudo python3 infra/sdn/mininet_service.py            # kernel datapath (default)
    sudo python3 infra/sdn/mininet_service.py --datapath user   # userspace OVS (no kernel module)

Topology (port numbers are fixed so the Ryu exporter can name links):

        s1 (RAN site: gNB n3 attaches here, 10.100.3.10)
     p2 /  \\ p1
       /    \\
 p1  s4      s2  p2
       \\    /
     p2 \\  / p1
        s3 (core site: UPF n3 attaches here, 10.100.3.20)

  link    clockwise end      counter-clockwise end
  s1-s2   s1 port 1          s2 port 2
  s2-s3   s2 port 1          s3 port 2
  s3-s4   s3 port 1          s4 port 2
  s4-s1   s4 port 1          s1 port 2

With the bridge priorities set in ryu_apps/stp_ring.py (s1 root, then s2 < s3 < s4), spanning tree
blocks s3's port toward s4, so N3 follows s1-s2-s3; taking s1-s2 or s2-s3 down reroutes it via s4
after reconvergence (~30 s with stplib's default forward delay of 15 s).

Impairments apply to BOTH ends of a link (both directions): a delay fault of d ms adds 2d ms to the
round-trip time, a loss of p % gives 1-(1-p)^2 round-trip loss. labels.csv records the one-way value.

Control socket (JSON lines, /run/testbed-mininet.sock):
  {"op": "impair", "a": "s1", "b": "s2", "delay_ms": 10, "jitter_ms": 2, "loss_pct": 0}
  {"op": "reset", "a": "s1", "b": "s2"}       {"op": "link", "a": .., "b": .., "state": "down"|"up"}
  {"op": "reset_all"}                          {"op": "status"}
Every reply carries "t_ms": the time right after the change returned on both link ends.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from functools import partial

from mininet.link import Link, TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

RING = [("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s1")]
BASE = {"bw": 1000, "delay": 0.5}           # Mb/s, ms one-way per link (a metro fibre hop)
SOCK = "/run/testbed-mininet.sock"
log = logging.getLogger("sdn")


class Transport:
    def __init__(self, datapath: str, controller_port: int, use_tc: bool = True):
        sw = partial(OVSSwitch, protocols="OpenFlow13", failMode="secure", datapath=datapath)
        # use_tc=False: plain links, for kernels without sch_netem (impair/reset then refuse)
        self.use_tc = use_tc
        self.net = Mininet(switch=sw, link=TCLink if use_tc else Link, controller=None,
                           autoSetMacs=False, build=False)
        self.net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=controller_port)
        for i in range(1, 5):
            self.net.addSwitch(f"s{i}", dpid=f"{i:016x}")
        self.links = {}
        for a, b in RING:
            kw = {"bw": BASE["bw"], "delay": f"{BASE['delay']}ms"} if use_tc else {}
            l = self.net.addLink(a, b, port1=1, port2=2, **kw)
            self.links[(a, b)] = l
        self.state = {f"{a}-{b}": {"state": "up", "impaired": False, "params": {}} for a, b in RING}
        self.lock = threading.Lock()

    def start(self):
        self.net.build()
        self.net.start()

    def _link(self, a, b):
        if (a, b) in self.links:
            return self.links[(a, b)], f"{a}-{b}"
        if (b, a) in self.links:
            return self.links[(b, a)], f"{b}-{a}"
        raise KeyError(f"no link {a}-{b}")

    def impair(self, a, b, delay_ms=0.0, jitter_ms=0.0, loss_pct=0.0):
        if not self.use_tc:
            raise RuntimeError("started with --no-tc: link impairments unavailable")
        link, name = self._link(a, b)
        kw = {"bw": BASE["bw"], "delay": f"{BASE['delay'] + float(delay_ms)}ms"}
        if jitter_ms:
            kw["jitter"] = f"{float(jitter_ms)}ms"
        if loss_pct:
            kw["loss"] = float(loss_pct)
        with self.lock:
            for intf in (link.intf1, link.intf2):
                intf.config(**kw)
            self.state[name].update(impaired=True, params={"delay_ms": delay_ms, "jitter_ms": jitter_ms,
                                                           "loss_pct": loss_pct})
        return int(time.time() * 1000)

    def reset(self, a, b):
        link, name = self._link(a, b)
        if not self.use_tc:
            return int(time.time() * 1000)
        with self.lock:
            for intf in (link.intf1, link.intf2):
                intf.config(bw=BASE["bw"], delay=f"{BASE['delay']}ms")
            self.state[name].update(impaired=False, params={})
        return int(time.time() * 1000)

    def set_link(self, a, b, state):
        _, name = self._link(a, b)
        with self.lock:
            self.net.configLinkStatus(name.split("-")[0], name.split("-")[1], state)
            self.state[name]["state"] = state
        return int(time.time() * 1000)

    def reset_all(self):
        for a, b in RING:
            self.set_link(a, b, "up")
            self.reset(a, b)
        return int(time.time() * 1000)

    def status(self):
        sws = [{"name": s.name, "connected": bool(s.connected())} for s in self.net.switches]
        links = [{"name": n, **v} for n, v in self.state.items()]
        return {"switches": sws, "links": links, "t_ms": int(time.time() * 1000)}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "impair":
            t = self.impair(req["a"], req["b"], req.get("delay_ms", 0), req.get("jitter_ms", 0),
                            req.get("loss_pct", 0))
        elif op == "reset":
            t = self.reset(req["a"], req["b"])
        elif op == "link":
            t = self.set_link(req["a"], req["b"], req["state"])
        elif op == "reset_all":
            t = self.reset_all()
        elif op == "status":
            return {"ok": True, **self.status()}
        else:
            raise ValueError(f"unknown op {op}")
        log.info("%s -> t_ms=%d", json.dumps(req, sort_keys=True), t)
        return {"ok": True, "t_ms": t}


def serve(tr: Transport, path: str, stop: threading.Event):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o666)       # single-user laptop: the injector runs as the normal user
    srv.listen(8)
    srv.settimeout(1.0)
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            try:
                resp = tr.handle(json.loads(buf.decode()))
            except Exception as e:      # report, never crash the transport
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            conn.sendall((json.dumps(resp) + "\n").encode())
    srv.close()
    os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", default="kernel", choices=["kernel", "user"])
    ap.add_argument("--controller-port", type=int, default=6653)
    ap.add_argument("--socket", default=SOCK)
    ap.add_argument("--no-tc", action="store_true", help="kernel without sch_netem: no impairments")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    setLogLevel("warning")
    tr = Transport(a.datapath, a.controller_port, use_tc=not a.no_tc)
    tr.start()
    stop = threading.Event()
    log.info("ring up; control socket %s (Ctrl+C to stop -- stop the 5G pods first)", a.socket)
    try:
        serve(tr, a.socket, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        tr.net.stop()


if __name__ == "__main__":
    main()
