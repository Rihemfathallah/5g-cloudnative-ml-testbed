"""LSTM handover predictor: features, labels, model, and the policy that uses it.

Input at time t (serving cell s), history of L samples:
  channel 0      filtered RSRP of s, scaled ((x + 100) / 20)
  channels 1..K  filtered RSRP of the K strongest neighbours NOW, minus that of s, / 10 dB
  channel K+1    UE speed / 120 km/h
Neighbour slots are ranked by their CURRENT filtered RSRP; the target head predicts a slot.

Label at (t, s), from the NOISE-FREE future (definition used in the report):
  y = 1 iff some neighbour n has rsrp_n - rsrp_s >= X dB continuously for >= D ms, with that
  sustained period STARTING in (t, t + H]. Target = the neighbour whose period starts first
  (ties: larger margin). States with t + H + D beyond the trace end are not labelled.
"""
from __future__ import annotations

import numpy as np

from testbed.handover.sim import Policy


def input_tensor(filt_hist: np.ndarray, serving: np.ndarray, speed: np.ndarray, K: int):
    """filt_hist: (B, L, C) filtered RSRP history; serving (B,); speed (B,) km/h.
    Returns x (B, L, K+2) float32 and the neighbour cell id of each slot (B, K)."""
    B, L, C = filt_hist.shape
    ar = np.arange(B)
    now = filt_hist[:, -1].copy()
    now[ar, serving] = -np.inf
    slots = np.argsort(-now, axis=1)[:, :K]                              # (B, K)
    s_hist = filt_hist[ar, :, serving]                                   # (B, L)
    n_hist = np.take_along_axis(filt_hist, slots[:, None, :].repeat(L, 1), axis=2)  # (B, L, K)
    x = np.concatenate([((s_hist + 100) / 20)[..., None], (n_hist - s_hist[..., None]) / 10,
                        np.repeat((speed / 120)[:, None, None], L, 1)], axis=2)
    return x.astype(np.float32), slots


def sustained_starts(rsrp_traj: np.ndarray, X: float, D: int) -> np.ndarray:
    """For one trajectory (T, C): start[s, n, t] = True if rsrp_n - rsrp_s >= X on [t, t+D)."""
    T, C = rsrp_traj.shape
    diff = rsrp_traj[:, None, :] - rsrp_traj[:, :, None]           # diff[t, s, n] = r[t,n] - r[t,s]
    good = (diff >= X).astype(np.int32)
    cs = np.concatenate([np.zeros((1, C, C), np.int32), np.cumsum(good, axis=0)])
    out = np.zeros((T, C, C), bool)
    if T >= D:
        out[: T - D + 1] = (cs[D:] - cs[: T - D + 1]) == D
    return np.transpose(out, (1, 2, 0))                            # (C_s, C_n, T)


def labels_for_states(rsrp: np.ndarray, traj: np.ndarray, t: np.ndarray, serving: np.ndarray,
                      X: float, H: int, D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns y (0/1), target cell (-1 if y=0) and a validity mask."""
    T = rsrp.shape[1]
    y = np.zeros(len(t), np.int64)
    tgt = np.full(len(t), -1)
    valid = t + H + D <= T
    cache: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(valid):
        k = int(traj[i])
        if k not in cache:
            st = sustained_starts(rsrp[k], X, D)
            # next_start[s, n, t] = first index >= t with st True, else large
            nxt = np.full(st.shape, 10**9, np.int64)
            run = np.full(st.shape[:2], 10**9, np.int64)
            for tt in range(T - 1, -1, -1):
                run = np.where(st[:, :, tt], tt, run)
                nxt[:, :, tt] = run
            cache[k] = nxt
        nxt = cache[k]
        s, ti = int(serving[i]), int(t[i])
        first = nxt[s, :, ti + 1] if ti + 1 < T else np.full(rsrp.shape[2], 10**9)
        first = first.copy()
        first[s] = 10**9
        if first.min() <= ti + H:
            cand = np.flatnonzero(first == first.min())
            margins = rsrp[k, first.min(), cand] - rsrp[k, first.min(), s]
            y[i], tgt[i] = 1, int(cand[np.argmax(margins)])
    return y, tgt, valid


class LSTMNet:
    def __new__(cls, n_in, hidden, layers, K):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
                self.ho = nn.Linear(hidden, 1)
                self.tgt = nn.Linear(hidden, K)

            def forward(self, x):
                y, _ = self.lstm(x)
                h = y[:, -1]
                return self.ho(h).squeeze(-1), self.tgt(h)
        return Net()


class LSTMPolicy(Policy):
    """Hand over when P(HO within H) > tau and the UE has stayed >= min_stay in its cell;
    the target is the slot with the highest target logit."""

    def __init__(self, net, filt: np.ndarray, traj: np.ndarray, speed: np.ndarray, L: int, K: int,
                 tau: np.ndarray, min_stay_s: np.ndarray, device="cpu"):
        self.net, self.filt, self.traj, self.speed = net, filt, traj, speed
        self.L, self.K = L, K
        self.tau, self.min_stay = np.asarray(tau, float), np.asarray(min_stay_s, float)

    def decide(self, t, serving, filt_t, can, since_ho_s):
        import torch
        B = len(serving)
        out = np.full(B, -1)
        if t < self.L - 1:
            return out
        act = can & (since_ho_s >= self.min_stay)
        if not act.any():
            return out
        idx = np.flatnonzero(act)
        hist = self.filt[self.traj[idx], t - self.L + 1: t + 1]
        x, slots = input_tensor(hist, serving[idx], self.speed[self.traj[idx]], self.K)
        with torch.no_grad():
            lo, tl = self.net(torch.from_numpy(x))
        p = torch.sigmoid(lo).numpy()
        slot = tl.argmax(dim=1).numpy()
        go = p > self.tau[idx]
        out[idx[go]] = slots[np.arange(len(idx)), slot][go]
        return out
