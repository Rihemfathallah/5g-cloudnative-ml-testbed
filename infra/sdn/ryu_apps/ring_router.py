"""Ryu app for the N3 ring: controller-computed loop-free forwarding + Prometheus exporter.

    ryu-manager infra/sdn/ryu_apps/ring_router.py        (Python 3.9 env, see infra/VERSIONS.md)

Design (ADR-3, revised after infra/sdn/check_ring.py):
* A learning switch on a ring loops broadcasts forever, so forwarding is restricted to a TREE.
* Ryu's stplib was tried first and rejected: after a failure on the active path it never moved the
  blocked port to FORWARD (s3 kept its stale root port), i.e. a link failure was a permanent
  blackout instead of a reroute. Spanning tree also delays every new pod port by 2 x 15 s.
* Instead the controller owns the tree: it knows the ring wiring, tracks link state from
  OFPPortStatus messages, and computes a BFS tree rooted at s1 (ties -> lower dpid). With all links
  up the tree is s1-s2, s2-s3, s1-s4, so N3 (gNB on s1, UPF on s3) takes s1-s2-s3 and the s3-s4
  link carries nothing. When a tree link goes down, the tree is recomputed at once, every learned
  flow is flushed, and traffic re-learns over s1-s4-s3. When it comes back, the original tree is
  restored the same way.
* Ports other than the two ring ports of a switch are EDGE ports (pod attachments made by OVS-CNI);
  they always forward.

Exported on :9300:
  ryu_port_{rx,tx}_bytes_total, ryu_port_{rx,tx}_packets_total, ryu_port_tx_dropped_total
      {link, dpid, port}   for the four inter-switch links, at each link's clockwise end
  ryu_port_stats{dpid, port, field}   every port
  ryu_link_forwarding{link}           1 if the link is up AND part of the forwarding tree
  ryu_link_up{link}                   1 if both ends report link up
  ryu_tree_changes_total              number of tree recomputations that changed the tree
"""
from __future__ import annotations

import threading
from collections import deque

from prometheus_client import make_wsgi_app
from prometheus_client.core import REGISTRY, CounterMetricFamily, GaugeMetricFamily
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, packet
from ryu.ofproto import ofproto_v1_3

ROOT = 1
# link name -> ((dpid, port) clockwise end, (dpid, port) counter-clockwise end); matches mininet_service.py
LINKS = {"s1-s2": ((1, 1), (2, 2)), "s2-s3": ((2, 1), (3, 2)),
         "s3-s4": ((3, 1), (4, 2)), "s4-s1": ((4, 1), (1, 2))}
RING_PORTS = {1, 2}
POLL_S = 2.0
EXPORT_PORT = 9300
FLOW_IDLE_S = 30


def bfs_tree(up_links: set[str]) -> set[str]:
    """Links of the BFS tree rooted at s1 over the links that are up; neighbours visited in
    increasing dpid order, so the tree is deterministic."""
    adj: dict[int, list[tuple[int, str]]] = {d: [] for d in range(1, 5)}
    for name in up_links:
        (a, _), (b, _) = LINKS[name]
        adj[a].append((b, name))
        adj[b].append((a, name))
    seen, tree, q = {ROOT}, set(), deque([ROOT])
    while q:
        u = q.popleft()
        for v, name in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                tree.add(name)
                q.append(v)
    return tree


class _Collector:
    def __init__(self, app):
        self.app = app

    def collect(self):
        with self.app.lock:
            stats, tree, up = dict(self.app.stats), set(self.app.tree), set(self.app.up_links)
            changes = self.app.tree_changes
        fams = {k: CounterMetricFamily(f"ryu_port_{k}", f"port {k} on ring links", labels=["link", "dpid", "port"])
                for k in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "tx_dropped")}
        allp = GaugeMetricFamily("ryu_port_stats", "raw port counters, all ports", labels=["dpid", "port", "field"])
        fwd = GaugeMetricFamily("ryu_link_forwarding", "link up and in the forwarding tree", labels=["link"])
        lup = GaugeMetricFamily("ryu_link_up", "both ends report link up", labels=["link"])
        tc = CounterMetricFamily("ryu_tree_changes", "forwarding tree recomputations that changed it")
        tc.add_metric([], changes)
        for (dpid, port), s in stats.items():
            for f, v in s.items():
                allp.add_metric([str(dpid), str(port), f], v)
        for name, (cw, _) in LINKS.items():
            if cw in stats:
                for k, fam in fams.items():
                    fam.add_metric([name, str(cw[0]), str(cw[1])], stats[cw][k])
            fwd.add_metric([name], 1.0 if (name in tree and name in up) else 0.0)
            lup.add_metric([name], 1.0 if name in up else 0.0)
        yield from fams.values()
        yield from (allp, fwd, lup, tc)


class RingRouter(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port: dict[int, dict[str, int]] = {}
        self.datapaths = {}
        self.stats: dict[tuple[int, int], dict] = {}
        self.port_up: dict[tuple[int, int], bool] = {}
        self.up_links: set[str] = set()
        self.tree: set[str] = set()
        self.tree_changes = 0
        self.lock = threading.Lock()
        REGISTRY.register(_Collector(self))
        # prometheus_client.start_http_server fails under Ryu: eventlet's monkey-patched
        # socket.getaddrinfo rejects its keyword arguments. Serve through eventlet instead.
        self.http = hub.WSGIServer(("0.0.0.0", EXPORT_PORT), make_wsgi_app())
        hub.spawn(self.http.serve_forever)
        hub.spawn(self._poll)

    # ---------------------------------------------------------------- topology
    def _recompute(self, reason: str) -> None:
        with self.lock:
            up = {n for n, (a, b) in LINKS.items() if self.port_up.get(a) and self.port_up.get(b)}
            new = bfs_tree(up)
            changed = new != self.tree
            self.up_links, self.tree = up, new
            if changed:
                self.tree_changes += 1
        if changed:
            self.logger.info("tree %s (%s); up links %s", sorted(new), reason, sorted(up))
            for dp in list(self.datapaths.values()):
                self._flush(dp)

    def _allowed_ports(self, dpid: int, all_ports: list[int]) -> set[int]:
        """Edge ports + ring ports whose link is in the tree."""
        allowed = {p for p in all_ports if p not in RING_PORTS}
        for name in self.tree:
            for (d, p) in LINKS[name]:
                if d == dpid:
                    allowed.add(p)
        return allowed

    # ---------------------------------------------------------------- flows
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        self._add_flow(dp, 0, parser.OFPMatch(), [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)])
        dp.send_msg(parser.OFPPortDescStatsRequest(dp, 0))

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def _port_desc_handler(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            for p in ev.msg.body:
                if p.port_no <= 0xffffff00:
                    self.port_up[(dpid, p.port_no)] = not (p.state & ev.msg.datapath.ofproto.OFPPS_LINK_DOWN) \
                        and not (p.config & ev.msg.datapath.ofproto.OFPPC_PORT_DOWN)
        self._recompute(f"ports of s{dpid}")

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        msg = ev.msg
        ofp = msg.datapath.ofproto
        p = msg.desc
        up = not (p.state & ofp.OFPPS_LINK_DOWN) and not (p.config & ofp.OFPPC_PORT_DOWN) \
            and msg.reason != ofp.OFPPR_DELETE
        with self.lock:
            self.port_up[(msg.datapath.id, p.port_no)] = up
        if p.port_no in RING_PORTS:
            self._recompute(f"s{msg.datapath.id} port {p.port_no} {'up' if up else 'down'}")

    def _add_flow(self, dp, priority, match, actions, idle=0):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst,
                                      idle_timeout=idle))

    def _flush(self, dp):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        dp.send_msg(parser.OFPFlowMod(datapath=dp, command=ofp.OFPFC_DELETE, table_id=ofp.OFPTT_ALL,
                                      out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY, priority=1,
                                      match=parser.OFPMatch()))
        # OFPFC_DELETE ignores priority: re-add the table-miss entry afterwards
        self._add_flow(dp, 0, parser.OFPMatch(), [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)])
        self.mac_to_port.pop(dp.id, None)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]
        ports = [p for (d, p) in self.port_up if d == dp.id and self.port_up[(d, p)]]
        allowed = self._allowed_ports(dp.id, ports)
        if in_port not in allowed:
            return                                   # arrived on a non-tree ring port: drop (no loops)
        eth = packet.Packet(msg.data).get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == 0x88cc:                   # LLDP: ignore
            return
        table = self.mac_to_port.setdefault(dp.id, {})
        table[eth.src] = in_port
        out = table.get(eth.dst)
        if out is not None and out in allowed:
            actions = [parser.OFPActionOutput(out)]
            self._add_flow(dp, 1, parser.OFPMatch(in_port=in_port, eth_dst=eth.dst), actions, idle=FLOW_IDLE_S)
        else:                                        # flood over the tree only
            actions = [parser.OFPActionOutput(p) for p in sorted(allowed) if p != in_port]
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                                        actions=actions, data=data))

    # ---------------------------------------------------------------- stats
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    def _poll(self):
        while True:
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(POLL_S)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        with self.lock:
            for s in ev.msg.body:
                if s.port_no > 0xffffff00:
                    continue
                self.stats[(dpid, s.port_no)] = {"rx_bytes": s.rx_bytes, "tx_bytes": s.tx_bytes,
                                                 "rx_packets": s.rx_packets, "tx_packets": s.tx_packets,
                                                 "tx_dropped": s.tx_dropped}
