"""Vectorised handover decision simulator, shared by every policy (ADR-7).

B agents step in lockstep over T samples. Agent b replays trajectory traj[b] with its own policy
parameters, so a whole parameter grid x all trajectories runs as one batch. Because A3 and the
LSTM go through exactly this code, any difference in results comes from the decision rule only.

Event model (stated in the report):
* Serving SINR is computed from noise-free large-scale RSRP under full load (all cells interfere).
* Radio link monitoring: SINR averaged (linear) over `qout_eval_ms`; when the average drops below
  Q_out, T310 starts; it stops if the average rises above Q_in; expiry = radio link failure (RLF).
* RLF -> outage of `reestablish_s`, then re-establishment on the strongest cell (true RSRP).
* Handover: decided on filtered measurements; the UE is interrupted for `ho_interruption_ms`;
  if the target SINR at execution is already below Q_out, it is a handover failure,
  counted as an RLF (it is the same user-visible event) and also counted separately.
* Ping-pong: a handover back to the cell that was left less than `ping_pong_s` earlier.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np

from testbed.handover.channel import noise_per_re_dbm


@dc.dataclass
class SimData:
    rsrp: np.ndarray      # (N, T, C) true large-scale RSRP
    filt: np.ndarray      # (N, T, C) L3-filtered measurements
    speed: np.ndarray     # (N,)
    dt_s: float


class Policy:
    """decide(t, serving, filt_t, can_decide) -> target cell per agent, -1 = no handover."""

    def reset(self, B: int, C: int) -> None: ...

    def on_handover(self, mask: np.ndarray) -> None: ...

    def decide(self, t: int, serving: np.ndarray, filt_t: np.ndarray, can: np.ndarray,
               since_ho_s: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class A3Policy(Policy):
    """TS 38.331 event A3 (offsets Ofn, Ocn, Ofp, Ocp = 0): Mn - Hys > Mp + Off held for TTT."""

    def __init__(self, hys_db: np.ndarray, ttt_ms: np.ndarray, off_db: float, dt_s: float):
        self.hys = np.asarray(hys_db, float)
        self.ttt_steps = np.maximum(1, np.round(np.asarray(ttt_ms) / (dt_s * 1000))).astype(int)
        self.off = off_db

    def reset(self, B, C):
        self.count = np.zeros((B, C), np.int32)

    def on_handover(self, mask):
        self.count[mask] = 0

    def decide(self, t, serving, filt_t, can, since_ho_s):
        B, C = filt_t.shape
        mp = filt_t[np.arange(B), serving][:, None]
        cond = (filt_t - self.hys[:, None] > mp + self.off)
        cond[np.arange(B), serving] = False
        self.count = np.where(cond, self.count + 1, 0)
        trig = (self.count >= self.ttt_steps[:, None]) & can[:, None]
        cand = np.where(trig, filt_t, -np.inf)
        tgt = np.argmax(cand, axis=1)
        return np.where(trig.any(axis=1), tgt, -1)


def simulate(data: SimData, traj: np.ndarray, policy: Policy, fcfg: dict, dep: dict,
             record_states: bool = False, state_stride: int = 1) -> dict:
    """Run B = len(traj) agents. Returns per-agent counters (and visited states if asked)."""
    rsrp, filt = data.rsrp, data.filt
    B, T, C = len(traj), rsrp.shape[1], rsrp.shape[2]
    dt = data.dt_s
    noise = 10 ** (noise_per_re_dbm(dep) / 10)
    q_out, q_in = 10 ** (fcfg["q_out_db"] / 10), 10 ** (fcfg["q_in_db"] / 10)
    win = max(1, int(round(fcfg["qout_eval_ms"] / 1000 / dt)))
    t310_steps = int(round(fcfg["t310_s"] / dt))
    reest_steps = int(round(fcfg["reestablish_s"] / dt))
    hoi_steps = int(round(fcfg["ho_interruption_ms"] / 1000 / dt))
    pp_steps = int(round(fcfg["ping_pong_s"] / dt))

    ar = np.arange(B)
    lin_all = 10 ** (rsrp[traj] / 10)                                # (B, T, C)
    serving = np.argmax(rsrp[traj, 0], axis=1)                      # attach to strongest
    prev_cell = np.full(B, -1)
    last_ho = np.full(B, -10**9)
    t310 = np.full(B, -1)                                           # -1 = not running
    blocked_until = np.zeros(B, int)                                # outage / interruption
    sinr_buf = np.zeros((B, win))
    n_ho = np.zeros(B, int); n_pp = np.zeros(B, int); n_rlf = np.zeros(B, int); n_hof = np.zeros(B, int)
    outage = np.zeros(B, int); sinr_sum_db = np.zeros(B); sinr_cnt = np.zeros(B, int)
    policy.reset(B, C)
    states = []

    def sinr_of(lin_t, cell):
        s = lin_t[ar, cell]
        return s / (lin_t.sum(axis=1) - s + noise)

    for t in range(T):
        lin_t = lin_all[:, t]
        connected = t >= blocked_until
        sinr = sinr_of(lin_t, serving)
        sinr_buf[:, t % win] = sinr
        avg = sinr_buf[:, : min(t + 1, win)].mean(axis=1)
        # outage: blocked (RLF re-establishment / HO interruption) or below Q_out
        outage += (~connected) | (avg < q_out)
        sinr_sum_db += np.where(connected, 10 * np.log10(sinr), 0)
        sinr_cnt += connected
        # ---- radio link monitoring
        start = connected & (t310 < 0) & (avg < q_out)
        t310[start] = t
        stop = (t310 >= 0) & (avg > q_in)
        t310[stop] = -1
        rlf = connected & (t310 >= 0) & (t - t310 >= t310_steps)
        if rlf.any():
            n_rlf += rlf
            blocked_until[rlf] = t + reest_steps
            new = np.argmax(rsrp[traj[rlf], min(T - 1, t + reest_steps)], axis=1)
            prev_cell[rlf] = -1          # a re-establishment is not a handover: no ping-pong
            serving[rlf] = new
            t310[rlf] = -1
            policy.on_handover(rlf)
        # ---- policy decision
        can = connected & ~rlf
        filt_t = filt[traj, t]
        if record_states and t % state_stride == 0:
            states.append((t, serving.copy(), can.copy()))
        tgt = policy.decide(t, serving, filt_t, can, (t - last_ho) * dt)
        do = (tgt >= 0) & can & (tgt != serving)
        if do.any():
            idx = np.flatnonzero(do)
            target_avg = sinr_of(lin_t, np.where(do, tgt, serving))[idx]
            fail = target_avg < q_out
            n_ho[idx] += 1
            pp = (prev_cell[idx] == tgt[idx]) & (t - last_ho[idx] <= pp_steps)
            n_pp[idx] += pp
            # failed HO -> RLF on the target side (counted in both n_hof and n_rlf)
            n_hof[idx[fail]] += 1
            n_rlf[idx[fail]] += 1
            prev_cell[idx] = serving[idx]
            serving[idx] = tgt[idx]
            last_ho[idx] = t
            blocked_until[idx] = t + np.where(fail, reest_steps, hoi_steps)
            t310[idx] = -1
            sinr_buf[idx] = sinr_of(lin_t, serving)[idx][:, None]
            policy.on_handover(do)

    minutes = T * dt / 60
    out = {"n_ho": n_ho, "n_pp": n_pp, "n_rlf": n_rlf, "n_hof": n_hof,
           "outage_frac": outage / T, "mean_sinr_db": sinr_sum_db / np.maximum(sinr_cnt, 1),
           "minutes": np.full(B, minutes)}
    if record_states:
        out["states"] = states
    return out


def rates(res: dict, mask: np.ndarray | None = None) -> dict:
    m = slice(None) if mask is None else mask
    mins = res["minutes"][m].sum()
    ho = res["n_ho"][m].sum()
    return {"ho_per_min": ho / mins, "pp_per_min": res["n_pp"][m].sum() / mins,
            "pp_ratio": res["n_pp"][m].sum() / max(ho, 1), "rlf_per_min": res["n_rlf"][m].sum() / mins,
            "hof_per_min": res["n_hof"][m].sum() / mins,
            "outage_frac": float(res["outage_frac"][m].mean()),
            "mean_sinr_db": float(res["mean_sinr_db"][m].mean()), "ue_minutes": float(mins)}


def objective(r: dict, w: dict) -> float:
    return w["w_rlf"] * r["rlf_per_min"] + w["w_pp"] * r["pp_per_min"] + w["w_ho"] * r["ho_per_min"]
