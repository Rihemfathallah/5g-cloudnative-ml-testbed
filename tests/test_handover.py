"""Channel model, L3 filter, A3, simulator events and labels."""
import numpy as np
import pytest

from testbed.handover.channel import (hex_sites, los_probability_uma, pathloss_uma_los,
                                      pathloss_uma_nlos, smooth_gaussian_along_path)
from testbed.handover.lstm import labels_for_states, sustained_starts
from testbed.handover.sim import A3Policy, Policy, SimData, rates, simulate
from testbed.handover.traces import l3_filter

DEP = {"h_ut_m": 1.5, "noise_figure_db": 9, "scs_khz": 30}
FC = {"q_out_db": -8, "q_in_db": -6, "qout_eval_ms": 200, "t310_s": 1.0, "reestablish_s": 0.5,
      "ho_interruption_ms": 40, "ping_pong_s": 1.0}


def test_pathloss_hand_values():
    # TR 38.901 Table 7.4.1-1, UMa LOS PL1 at d3D: 28 + 22 log10(d3D) + 20 log10(fc)
    d3d = np.sqrt(100**2 + 23.5**2)
    assert pathloss_uma_los(np.array(100.0), 25, 1.5, 3.5) == pytest.approx(28 + 22 * np.log10(d3d) + 20 * np.log10(3.5))
    assert (pathloss_uma_nlos(np.linspace(20, 2000, 50), 25, 1.5, 3.5) >=
            pathloss_uma_los(np.linspace(20, 2000, 50), 25, 1.5, 3.5)).all()
    # see tests/crosscheck_ns3.py: identical to ns-3's ThreeGppUmaPropagationLossModel


def test_los_probability():
    assert los_probability_uma(np.array(10.0)) == 1.0
    p = los_probability_uma(np.array([50.0, 200.0, 1000.0]))
    assert (np.diff(p) < 0).all() and 0 < p[-1] < 0.05


def test_hex_sites():
    s = hex_sites(7, 500)
    assert np.allclose(np.linalg.norm(s[1:], axis=1), 500)


def test_smooth_field_is_unit_variance_and_smooth():
    rng = np.random.default_rng(0)
    g = smooth_gaussian_along_path(np.full(20000, 0.5), 50, rng, 4)
    assert g.std() == pytest.approx(1.0, abs=0.15)
    assert np.abs(np.diff(g, axis=0)).max() < 0.1          # 0.5 m steps, 50 m correlation


def test_l3_filter_keeps_the_200ms_time_constant():
    """Step response at 40 ms sampling must match the specified filter at 200 ms sampling."""
    m40 = np.r_[np.zeros(1), np.ones(50)][None, :, None].repeat(1, 0)
    m200 = np.r_[np.zeros(1), np.ones(10)][None, :, None]
    f40 = l3_filter(m40, 4, 40)[0, :, 0]
    f200 = l3_filter(m200, 4, 200)[0, :, 0]
    assert f40[5] == pytest.approx(f200[1], rel=1e-9)      # 200 ms after the step
    assert f40[50] == pytest.approx(f200[10], rel=1e-9)


def _flat(T, C, serving_best=0, levels=None):
    lv = np.array(levels if levels is not None else [-80.0] + [-100.0] * (C - 1))
    return np.repeat(lv[None, None, :], T, axis=1).astype(np.float32)


def test_a3_triggers_exactly_after_ttt():
    T, C = 100, 3
    r = _flat(T, C, levels=[-80, -95, -95])
    r[0, 30:, 1] = -70.0                    # neighbour 1 becomes 10 dB better at t=30
    data = SimData(r, r.copy(), np.array([30.0]), 0.04)

    class Spy(A3Policy):
        fired = []

        def decide(self, t, serving, filt_t, can, since):
            out = super().decide(t, serving, filt_t, can, since)
            if (out >= 0).any():
                Spy.fired.append(t)
            return out
    pol = Spy(np.array([3.0]), np.array([160.0]), 0.0, 0.04)   # TTT = 4 samples
    res = simulate(data, np.array([0]), pol, FC, DEP)
    assert Spy.fired[0] == 30 + 4 - 1 and res["n_ho"][0] == 1


class Script(Policy):
    """Hands over to fixed targets at fixed steps."""
    def __init__(self, plan):
        self.plan = plan

    def decide(self, t, serving, filt_t, can, since):
        out = np.full(len(serving), -1)
        if t in self.plan:
            out[:] = self.plan[t]
        return out


def test_ping_pong_and_rlf_counting():
    T, C = 200, 2
    r = _flat(T, C, levels=[-80, -82])
    data = SimData(r, r, np.array([3.0]), 0.04)
    # 0 -> 1 at t=10, back to 0 at t=30 (0.8 s later) = ping-pong; back to 1 at t=100 (2.8 s) = not
    res = simulate(data, np.array([0]), Script({10: 1, 30: 0, 100: 1}), FC, DEP)
    assert res["n_ho"][0] == 3 and res["n_pp"][0] == 1
    # serving SINR far below Q_out from t=50: RLF after the 200 ms window fills + T310 (1 s)
    r2 = _flat(T, C, levels=[-80, -81])          # UE attaches to cell 0 (strongest at t=0)
    r2[0, 50:, 0] = -95.0
    r2[0, 50:, 1] = -80.0
    res2 = simulate(SimData(r2, r2, np.array([3.0]), 0.04), np.array([0]), Script({}), FC, DEP)
    assert res2["n_rlf"][0] == 1
    rt = rates(res2)
    assert rt["rlf_per_min"] == pytest.approx(1 / (T * 0.04 / 60))


def test_labels_sustained_future_margin():
    T, C = 100, 3
    r = np.full((1, T, C), -90.0)
    r[0, :, 0] = -80.0
    r[0, 40:, 2] = -75.0                    # cell 2 beats serving 0 by 5 dB from t=40 on
    st = sustained_starts(r[0], X=3.0, D=5)
    assert st[0, 2, 40] and not st[0, 2, 39] and not st[0, 1].any()
    y, tgt, valid = labels_for_states(r, np.zeros(3, int), np.array([20, 30, 38]), np.zeros(3, int),
                                      X=3.0, H=12, D=5)
    # t=20: next 12 samples are 21..32 -> no; t=30: 31..42 contains 40 -> yes; t=38 -> yes
    assert y.tolist() == [0, 1, 1] and tgt.tolist() == [-1, 2, 2] and valid.all()
