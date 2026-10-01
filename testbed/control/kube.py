"""Thin kubectl wrapper. Shelling out to kubectl (rather than the Python client) keeps the
dependency list short and makes every action reproducible by copy-pasting the logged command."""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time

log = logging.getLogger(__name__)


class KubeError(RuntimeError):
    pass


class Kube:
    def __init__(self, namespace: str = "5g", kubectl: str = "kubectl", dry_run: bool = False):
        self.ns = namespace
        self.kubectl = kubectl
        self.dry_run = dry_run
        self.history: list[str] = []

    def _run(self, args: list[str], stdin: str | None = None, timeout: float = 60,
             check: bool = True) -> str:
        cmd = [self.kubectl, *args]
        self.history.append(shlex.join(cmd))
        log.debug("kubectl: %s", shlex.join(cmd))
        if self.dry_run:
            return ""
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode != 0:
            raise KubeError(f"{shlex.join(cmd)} failed: {p.stderr.strip()}")
        return p.stdout

    def apply(self, manifest: str) -> None:
        self._run(["apply", "-f", "-"], stdin=manifest)

    def delete(self, kind: str, name: str, wait: bool = True) -> None:
        self._run(["-n", self.ns, "delete", kind, name, "--ignore-not-found",
                   f"--wait={'true' if wait else 'false'}"], timeout=120)

    def get_json(self, *args: str) -> dict:
        out = self._run(["-n", self.ns, "get", *args, "-o", "json"])
        return json.loads(out) if out else {}

    def exec(self, pod_selector: str, container: str, command: list[str],
             timeout: float = 600, check: bool = True) -> str:
        pod = self.pod_name(pod_selector)
        return self._run(["-n", self.ns, "exec", pod, "-c", container, "--", *command],
                         timeout=timeout, check=check)

    def pod_name(self, selector: str) -> str:
        if self.dry_run:
            return f"{selector.split('=')[-1]}-dry"
        out = self._run(["-n", self.ns, "get", "pods", "-l", selector,
                         "--field-selector=status.phase=Running", "-o",
                         "jsonpath={.items[0].metadata.name}"])
        if not out:
            raise KubeError(f"no running pod for selector {selector}")
        return out

    def pods_ready(self, selector: str) -> bool:
        if self.dry_run:
            return True
        doc = self.get_json("pods", "-l", selector)
        items = doc.get("items", [])
        if not items:
            return False
        for pod in items:
            if pod["metadata"].get("deletionTimestamp"):
                return False
            conds = {c["type"]: c["status"] for c in pod.get("status", {}).get("conditions", [])}
            if conds.get("Ready") != "True":
                return False
        return True

    def wait_ready(self, selector: str, timeout_s: float = 300, poll_s: float = 0.5,
                   must_change_from: set[str] | None = None) -> float:
        """Wait until all pods matching `selector` are Ready. With `must_change_from`, also wait
        until none of the original pod names remains (a killed pod must be REPLACED)."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.dry_run:
                return time.time()
            names = set()
            if must_change_from:
                doc = self.get_json("pods", "-l", selector)
                names = {p["metadata"]["name"] for p in doc.get("items", [])}
            if self.pods_ready(selector) and not (names & (must_change_from or set())):
                return time.time()
            time.sleep(poll_s)
        raise KubeError(f"pods {selector} not Ready after {timeout_s}s")

    def condition_true(self, kind: str, name: str, cond_type: str) -> bool:
        if self.dry_run:
            return True
        doc = self.get_json(kind, name)
        for c in doc.get("status", {}).get("conditions", []) or []:
            if c.get("type") == cond_type and c.get("status") == "True":
                return True
        return False
