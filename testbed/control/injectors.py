"""Fault injectors. Exactly one injection point per fault type (ADR-4):

  delay, loss, link_down  -> Mininet links of the SDN ring          (MininetInjector)
  cpu, mem, pod_kill      -> Chaos Mesh custom resources             (ChaosMeshInjector)
  flood                   -> parallel iperf3 from the UE pod         (TrafficFloodInjector)

Contract: `apply()` returns the time (Unix ms) at which the impairment was CONFIRMED active,
`remove()` the time at which it was confirmed gone. Those times -- not the plan -- become
t_applied / t_removed in labels.csv (ADR-6).
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod

import yaml

from testbed.common.clock import RealClock, SimClock
from testbed.common.config import parse_size_bytes
from testbed.control.kube import Kube, KubeError
from testbed.control.schedule import FaultPlan

log = logging.getLogger(__name__)


class InjectionError(RuntimeError):
    pass


class Injector(ABC):
    def __init__(self, clock):
        self.clock = clock

    @abstractmethod
    def apply(self, plan: FaultPlan, run_id: str) -> int: ...

    @abstractmethod
    def remove(self, plan: FaultPlan, run_id: str) -> int: ...

    def hold(self, plan: FaultPlan, stop: threading.Event | None) -> None:
        """Keep the impairment active for its planned duration."""
        self.clock.sleep(plan.duration_s, stop)

    def cleanup_all(self, run_id: str) -> None:
        """Remove every impairment this injector may have left behind (crash / Ctrl+C)."""


def _link_of(target: str) -> tuple[str, str]:
    if not target.startswith("link:"):
        raise InjectionError(f"mininet injector needs a link target, got {target}")
    a, b = target[5:].split("-")
    return a, b


def _pod_of(target: str) -> str:
    if not target.startswith("pod:"):
        raise InjectionError(f"expected pod:<app> target, got {target}")
    return target[4:]


# ------------------------------------------------------------------------------------ Mininet
class MininetRPC:
    """JSON-lines client for infra/sdn/mininet_service.py (runs as root next to Mininet)."""

    def __init__(self, sock_path: str = "/run/testbed-mininet.sock", timeout: float = 10):
        self.sock_path = sock_path
        self.timeout = timeout

    def call(self, **req) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect(self.sock_path)
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        resp = json.loads(buf.decode() or "{}")
        if not resp.get("ok"):
            raise InjectionError(f"mininet rpc {req.get('op')} failed: {resp.get('error')}")
        return resp


class MininetInjector(Injector):
    def __init__(self, clock, rpc: MininetRPC | None = None):
        super().__init__(clock)
        self.rpc = rpc or MininetRPC()

    def apply(self, plan: FaultPlan, run_id: str) -> int:
        a, b = _link_of(plan.target)
        p = plan.params
        if plan.type == "delay":
            r = self.rpc.call(op="impair", a=a, b=b, delay_ms=p["delay_ms"],
                              jitter_ms=p.get("jitter_ms", 0), loss_pct=0)
        elif plan.type == "loss":
            r = self.rpc.call(op="impair", a=a, b=b, delay_ms=0, jitter_ms=0,
                              loss_pct=p["loss_pct"])
        elif plan.type == "link_down":
            r = self.rpc.call(op="link", a=a, b=b, state="down")
        else:
            raise InjectionError(f"mininet cannot inject {plan.type}")
        # The service returns the time right after `tc`/`ip link` returned on BOTH link ends.
        return int(r["t_ms"])

    def remove(self, plan: FaultPlan, run_id: str) -> int:
        a, b = _link_of(plan.target)
        if plan.type == "link_down":
            r = self.rpc.call(op="link", a=a, b=b, state="up")
        else:
            r = self.rpc.call(op="reset", a=a, b=b)
        return int(r["t_ms"])

    def cleanup_all(self, run_id: str) -> None:
        try:
            self.rpc.call(op="reset_all")
        except Exception as e:  # pragma: no cover - best effort during teardown
            log.error("mininet reset_all failed: %s", e)


# ------------------------------------------------------------------------------------ Chaos Mesh
def chaos_manifest(plan: FaultPlan, run_id: str, namespace: str, safety_margin_s: float = 120) -> dict:
    """Render the Chaos Mesh resource for a plan.

    `duration` is a SAFETY bound (planned + margin): the injector deletes the resource itself, but
    if the injector dies the stress still ends on its own instead of polluting hours of data.
    """
    app = _pod_of(plan.target)
    name = f"{run_id[-24:].lower().replace('_', '-')}-{plan.fault_id}".strip("-")
    meta = {"name": name, "namespace": namespace,
            "labels": {"testbed/run": run_id[-63:].replace(":", "-"), "testbed/fault": plan.fault_id}}
    selector = {"namespaces": [namespace], "labelSelectors": {"app": app}}
    safety = f"{int(plan.duration_s + safety_margin_s)}s"
    if plan.type == "cpu":
        return {"apiVersion": "chaos-mesh.org/v1alpha1", "kind": "StressChaos", "metadata": meta,
                "spec": {"mode": "all", "selector": selector, "duration": safety,
                         "stressors": {"cpu": {"workers": int(plan.params["workers"]),
                                               "load": int(plan.params["load"])}}}}
    if plan.type == "mem":
        size = parse_size_bytes(plan.params["size"])
        return {"apiVersion": "chaos-mesh.org/v1alpha1", "kind": "StressChaos", "metadata": meta,
                "spec": {"mode": "all", "selector": selector, "duration": safety,
                         "stressors": {"memory": {"workers": 1, "size": f"{size // 1024**2}MB"}}}}
    if plan.type == "pod_kill":
        return {"apiVersion": "chaos-mesh.org/v1alpha1", "kind": "PodChaos", "metadata": meta,
                "spec": {"action": "pod-kill", "mode": "one", "selector": selector,
                         "gracePeriod": 0}}
    raise InjectionError(f"chaos-mesh cannot inject {plan.type}")


class ChaosMeshInjector(Injector):
    def __init__(self, clock, kube: Kube, confirm_timeout_s: float = 60):
        super().__init__(clock)
        self.kube = kube
        self.confirm_timeout_s = confirm_timeout_s
        self._replaced_from: dict[str, set[str]] = {}

    def _kind_name(self, plan, run_id):
        m = chaos_manifest(plan, run_id, self.kube.ns)
        return m["kind"].lower(), m["metadata"]["name"], m

    def apply(self, plan: FaultPlan, run_id: str) -> int:
        kind, name, m = self._kind_name(plan, run_id)
        app = _pod_of(plan.target)
        if plan.type == "pod_kill" and not self.kube.dry_run:
            doc = self.kube.get_json("pods", "-l", f"app={app}")
            self._replaced_from[plan.fault_id] = {p["metadata"]["name"] for p in doc.get("items", [])}
        self.kube.apply(yaml.safe_dump(m))
        deadline = time.time() + self.confirm_timeout_s
        while time.time() < deadline:
            if self.kube.condition_true(kind, name, "AllInjected"):
                return self.clock.now_ms()
            time.sleep(0.5)
        self.kube.delete(kind, name, wait=False)
        raise InjectionError(f"{kind}/{name} never reported AllInjected")

    def hold(self, plan: FaultPlan, stop) -> None:
        if plan.type == "pod_kill":
            return  # one-shot: the 'effect' lasts until the replacement pod is Ready (in remove)
        super().hold(plan, stop)

    def remove(self, plan: FaultPlan, run_id: str) -> int:
        kind, name, _ = self._kind_name(plan, run_id)
        if plan.type == "pod_kill":
            app = _pod_of(plan.target)
            self.kube.wait_ready(f"app={app}", timeout_s=300,
                                 must_change_from=self._replaced_from.pop(plan.fault_id, None))
            t = self.clock.now_ms()
            self.kube.delete(kind, name, wait=True)
            # settle: the NF must re-register with the NRF and the gNB must reconnect; the
            # planned duration is spent here so the next gap starts after recovery
            self.clock.sleep(plan.duration_s)
            return t
        self.kube.delete(kind, name, wait=True)
        # Deletion returns when the resource is gone, i.e. Chaos Mesh has recovered the target.
        return self.clock.now_ms()

    def cleanup_all(self, run_id: str) -> None:
        label = f"testbed/run={run_id[-63:].replace(':', '-')}"
        for kind in ("stresschaos", "podchaos"):
            try:
                self.kube._run(["-n", self.kube.ns, "delete", kind, "-l", label,
                                "--ignore-not-found"], check=False)
            except Exception as e:  # pragma: no cover
                log.error("chaos cleanup failed: %s", e)


# ------------------------------------------------------------------------------------ traffic flood
class TrafficFloodInjector(Injector):
    """Many parallel iperf3 streams from the UE probe container toward the data network."""

    def __init__(self, clock, kube: Kube, server: str = "10.45.0.1", dev: str = "tun_srsue"):
        super().__init__(clock)
        self.kube, self.server, self.dev = kube, server, dev
        self._proc: subprocess.Popen | None = None

    def apply(self, plan: FaultPlan, run_id: str) -> int:
        p = plan.params
        cmd = ["iperf3", "-c", self.server, "--bind-dev", self.dev, "-u",
               "-b", f"{p['mbps'] / p['parallel']:.2f}M", "-P", str(p["parallel"]),
               "-t", str(int(plan.duration_s) + 5), "-p", "5202", "--logfile", "/tmp/flood.log"]
        if self.kube.dry_run:
            return self.clock.now_ms()
        pod = self.kube.pod_name("app=ue")
        self._proc = subprocess.Popen([self.kube.kubectl, "-n", self.kube.ns, "exec", pod, "-c",
                                       "probe", "--", *cmd])
        time.sleep(1.0)  # iperf3 handshake; then traffic is flowing
        if self._proc.poll() not in (None, 0):
            raise InjectionError("flood iperf3 exited immediately")
        return self.clock.now_ms()

    def remove(self, plan: FaultPlan, run_id: str) -> int:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            if not self.kube.dry_run:
                self.kube.exec("app=ue", "probe", ["pkill", "-f", "iperf3 -c .* -p 5202"], check=False)
        return self.clock.now_ms()

    def cleanup_all(self, run_id: str) -> None:
        self.remove(None, run_id)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------ simulated
class SimInjector(Injector):
    """For the synthetic stand-in and tests: models apply/remove latency deterministically."""

    APPLY_LATENCY_MS = {"delay": 150, "loss": 150, "link_down": 100, "cpu": 2500, "mem": 2500,
                        "pod_kill": 1500, "flood": 1000}
    RECOVER_MS = {"pod_kill": 25000}  # time for the replacement pod to become Ready

    def __init__(self, clock: SimClock):
        super().__init__(clock)
        self.active: list[FaultPlan] = []

    def apply(self, plan, run_id):
        self.clock.sleep(self.APPLY_LATENCY_MS[plan.type] / 1000)
        self.active.append(plan)
        return self.clock.now_ms()

    def hold(self, plan, stop):
        if plan.type == "pod_kill":
            return
        super().hold(plan, stop)

    def remove(self, plan, run_id):
        if plan.type == "pod_kill":
            self.clock.sleep(self.RECOVER_MS["pod_kill"] / 1000)
            t = self.clock.now_ms()
            self.active.remove(plan)
            self.clock.sleep(plan.duration_s)
            return t
        self.clock.sleep(0.2)
        self.active.remove(plan)
        return self.clock.now_ms()
