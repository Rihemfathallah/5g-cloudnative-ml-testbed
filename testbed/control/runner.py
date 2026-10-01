"""Executes fault plans and traffic episodes, writing ground truth and context as it goes.

Rows are appended and flushed one by one, so a run that crashes after 3 hours still leaves
3 hours of usable, correctly labelled data.
"""
from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from typing import Callable

from testbed.common.runinfo import append_jsonl
from testbed.common.schema import LABEL_COLUMNS
from testbed.control.injectors import Injector
from testbed.control.schedule import FaultPlan, TrafficEpisode

log = logging.getLogger(__name__)


class LabelWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(LABEL_COLUMNS)

    def write(self, plan: FaultPlan, t_applied: int, t_removed: int, status: str,
              extra: dict | None = None) -> None:
        params = dict(plan.params)
        if extra:
            params.update(extra)
        row = [plan.fault_id, plan.type, plan.severity, json.dumps(params, sort_keys=True),
               plan.target, plan.injector, int(t_applied), int(t_removed), status]
        with self._lock, open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


class FaultRunner:
    def __init__(self, plans: list[FaultPlan], injectors: dict[str, Injector], clock,
                 labels: LabelWriter, run_id: str, grace_s: float, end_ms: int,
                 stop: threading.Event | None = None,
                 on_event: Callable[[str, FaultPlan, int], None] | None = None):
        self.plans, self.injectors, self.clock = plans, injectors, clock
        self.labels, self.run_id, self.grace_s, self.end_ms = labels, run_id, grace_s, end_ms
        self.stop = stop or threading.Event()
        self.on_event = on_event
        self.executed: list[dict] = []

    def _emit(self, what: str, plan: FaultPlan, t: int) -> None:
        if self.on_event:
            self.on_event(what, plan, t)

    def run(self) -> None:
        prev_effect_end = self.clock.now_ms()
        for plan in self.plans:
            if self.stop.is_set():
                break
            start_at = max(self.clock.now_ms(), prev_effect_end) + int(plan.gap_before_s * 1000)
            if start_at + int((plan.duration_s + self.grace_s) * 1000) > self.end_ms:
                log.info("no time left for %s; stopping fault schedule", plan.fault_id)
                break
            if not self.clock.sleep((start_at - self.clock.now_ms()) / 1000, self.stop):
                break
            inj = self.injectors[plan.injector]
            t_req = self.clock.now_ms()
            try:
                t_app = inj.apply(plan, self.run_id)
                self._emit("applied", plan, t_app)
                inj.hold(plan, self.stop)
                t_rem = inj.remove(plan, self.run_id)
                self._emit("removed", plan, t_rem)
                status = "ok"
            except Exception as e:
                log.exception("fault %s failed: %s", plan.fault_id, e)
                try:
                    inj.remove(plan, self.run_id)
                except Exception:
                    log.exception("removal after failure also failed")
                t_app, t_rem, status = t_req, self.clock.now_ms(), "failed"
            self.labels.write(plan, t_app, t_rem, status,
                              extra={"t_requested": t_req, "apply_latency_ms": t_app - t_req})
            self.executed.append({"fault_id": plan.fault_id, "status": status,
                                  "t_applied": t_app, "t_removed": t_rem})
            # Even a failed injection gets the full grace + gap: we do not know what it did.
            prev_effect_end = t_rem + int(self.grace_s * 1000)


class TrafficRunner:
    """Plays traffic episodes until stopped. `actions` maps episode kind -> callable(episode)
    that blocks for the episode's duration (real) or advances the clock (simulated)."""

    def __init__(self, episodes, actions: dict[str, Callable[[TrafficEpisode], None]], clock,
                 context_path: Path, end_ms: int, stop: threading.Event | None = None):
        self.episodes, self.actions, self.clock = episodes, actions, clock
        self.context_path, self.end_ms = context_path, end_ms
        self.stop = stop or threading.Event()

    def run(self) -> None:
        for ep in self.episodes:
            if self.stop.is_set() or self.clock.now_ms() >= self.end_ms:
                break
            t0 = self.clock.now_ms()
            try:
                self.actions[ep.kind](ep)
                ok = True
            except Exception as e:
                log.warning("traffic episode %s failed: %s", ep.kind, e)
                ok = False
            append_jsonl(self.context_path, {"kind": ep.kind, "t_start": t0,
                                             "t_end": self.clock.now_ms(), "ok": ok, **ep.params})
