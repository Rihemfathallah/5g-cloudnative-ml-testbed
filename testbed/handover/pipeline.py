"""RQ2 pipeline:  traces -> tune A3 per speed -> train LSTM (+ DAgger) -> tune tau/min-stay
per speed -> evaluate both on held-out test trajectories with paired bootstrap CIs.

Fairness rules (the reason this file exists):
* both policies are tuned on the SAME validation trajectories with the SAME objective J, per speed
* both are evaluated on the SAME test trajectories through the SAME simulator
* the LSTM's state distribution comes from A3 rollouts, then one DAgger round of its own rollouts,
  so it is not only trained on situations a different policy would visit
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from testbed.common.runinfo import freeze_manifest, seed_everything
from testbed.handover import traces as tr
from testbed.handover.lstm import LSTMNet, LSTMPolicy, input_tensor, labels_for_states
from testbed.handover.sim import A3Policy, SimData, objective, rates, simulate

log = logging.getLogger(__name__)


def _fast(cfg: dict) -> dict:
    import copy
    c = copy.deepcopy(cfg)
    c["mobility"].update(duration_s=40, n_traj={"train": 16, "val": 6, "test": 8})
    c["lstm"].update(epochs=3, dagger_iters=1, tau_grid=[0.4, 0.6, 0.8], min_stay_ms=[0, 500])
    c["a3"].update(hysteresis_db=[0, 2, 4, 6], ttt_ms=[40, 160, 640])
    c["seeds"] = [0]
    return c


def load_sim(cfg: dict, trace_dir: Path, split: str, speed: float) -> SimData:
    z = tr.load(trace_dir / f"{split}_{int(speed)}kmh.npz")
    m = cfg["measurement"]
    filt = tr.l3_filter(z["meas"], m["l3_k"], cfg["mobility"]["sample_ms"], m["l3_ref_period_ms"])
    return SimData(z["rsrp"], filt.astype(np.float32), z["speed"], cfg["mobility"]["sample_ms"] / 1000)


# ----------------------------------------------------------------------------- A3
def tune_a3(cfg, data: SimData) -> tuple[pd.DataFrame, dict]:
    grid = list(itertools.product(cfg["a3"]["hysteresis_db"], cfg["a3"]["ttt_ms"]))
    N = len(data.speed)
    traj = np.tile(np.arange(N), len(grid))
    hys = np.repeat([g[0] for g in grid], N)
    ttt = np.repeat([g[1] for g in grid], N)
    res = simulate(data, traj, A3Policy(hys, ttt, cfg["a3"]["offset_db"], data.dt_s),
                   cfg["failures"], cfg["deployment"])
    rows = []
    for k, (h, tt) in enumerate(grid):
        r = rates(res, slice(k * N, (k + 1) * N))
        rows.append({"hysteresis_db": h, "ttt_ms": tt, **r, "J": objective(r, cfg["objective"])})
    df = pd.DataFrame(rows)
    best = df.loc[df["J"].idxmin()]
    out = {"hysteresis_db": float(best["hysteresis_db"]), "ttt_ms": float(best["ttt_ms"])}
    out["on_grid_edge"] = _edge(out, cfg["a3"]["hysteresis_db"], cfg["a3"]["ttt_ms"])
    return df, out


def _edge(best: dict, hys_grid, ttt_grid) -> bool:
    """An optimum on the upper edge of a tuning grid means the baseline is not really tuned: the
    comparison would be biased toward the challenger. Warn loudly."""
    edge = best["hysteresis_db"] == max(hys_grid) or best["ttt_ms"] == max(ttt_grid)
    if edge:
        log.warning("A3 optimum %s is on the upper edge of the grid -- extend the grid", best)
    return bool(edge)


def a3_policy(cfg, best: dict, B: int, dt: float) -> A3Policy:
    return A3Policy(np.full(B, best["hysteresis_db"]), np.full(B, best["ttt_ms"]),
                    cfg["a3"]["offset_db"], dt)


# ----------------------------------------------------------------------------- LSTM data
def collect(cfg, data: SimData, policy_fn, stride: int):
    N = len(data.speed)
    traj = np.arange(N)
    res = simulate(data, traj, policy_fn(N), cfg["failures"], cfg["deployment"],
                   record_states=True, state_stride=stride)
    ts, tj, sv = [], [], []
    for t, serving, can in res["states"]:
        m = can
        ts.append(np.full(m.sum(), t)); tj.append(traj[m]); sv.append(serving[m])
    return np.concatenate(ts), np.concatenate(tj), np.concatenate(sv)


def make_xy(cfg, data: SimData, t, traj, serving):
    lc = cfg["lstm"]
    dt_ms = cfg["mobility"]["sample_ms"]
    L = int(round(lc["history_ms"] / dt_ms))
    lab = cfg["label"]
    H, D = int(round(lab["horizon_ms"] / dt_ms)), int(round(lab["sustain_ms"] / dt_ms))
    keep = t >= L - 1
    t, traj, serving = t[keep], traj[keep], serving[keep]
    y, tgt, valid = labels_for_states(data.rsrp, traj, t, serving, lab["margin_db"], H, D)
    t, traj, serving, y, tgt = t[valid], traj[valid], serving[valid], y[valid], tgt[valid]
    hist = np.stack([data.filt[k, tt - L + 1: tt + 1] for k, tt in zip(traj, t)])
    x, slots = input_tensor(hist, serving, data.speed[traj], lc["n_neigh"])
    slot = np.full(len(y), -1)
    for j in range(lc["n_neigh"]):
        slot[(slots[:, j] == tgt) & (y == 1)] = j
    return x, y.astype(np.float32), slot


def train_net(cfg, x, y, slot, xv, yv, slotv, seed: int):
    import torch
    seed_everything(seed)
    lc = cfg["lstm"]
    net = LSTMNet(x.shape[2], lc["hidden"], lc["layers"], lc["n_neigh"])
    opt = torch.optim.Adam(net.parameters(), lr=lc["lr"])
    pos = max(1.0, float(y.sum()))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    X, Y, S = map(torch.from_numpy, (x, y, slot.astype(np.int64)))
    Xv, Yv, Sv = map(torch.from_numpy, (xv, yv, slotv.astype(np.int64)))
    g = torch.Generator().manual_seed(seed)
    best, state, bad, hist = np.inf, None, 0, []

    def loss_of(xb, yb, sb):
        lo, tl = net(xb)
        l = bce(lo, yb)
        if (sb >= 0).any():
            l = l + ce(tl, sb)
        return l
    for ep in range(lc["epochs"]):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), lc["batch_size"]):
            b = perm[i:i + lc["batch_size"]]
            opt.zero_grad()
            l = loss_of(X[b], Y[b], S[b])
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += l.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(loss_of(Xv, Yv, Sv))
        hist.append({"epoch": ep, "train_loss": tot / len(X), "val_loss": vl})
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= lc["patience"]:
                break
    net.load_state_dict(state)
    net.eval()
    return net, hist


def classification_metrics(net, x, y, slot) -> dict:
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    with torch.no_grad():
        lo, tl = net(torch.from_numpy(x))
    p = torch.sigmoid(lo).numpy()
    out = {"n_states": int(len(y)), "pos_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    m = slot >= 0
    if m.any():
        out["target_accuracy"] = float((tl.argmax(1).numpy()[m] == slot[m]).mean())
    return out


def lstm_policy_factory(cfg, net, data: SimData):
    lc = cfg["lstm"]
    L = int(round(lc["history_ms"] / cfg["mobility"]["sample_ms"]))

    def make(traj, tau, stay_s):
        return LSTMPolicy(net, data.filt, traj, data.speed, L, lc["n_neigh"], tau, stay_s)
    return make


def tune_lstm(cfg, net, data: SimData) -> tuple[pd.DataFrame, dict]:
    lc = cfg["lstm"]
    grid = list(itertools.product(lc["tau_grid"], lc["min_stay_ms"]))
    N = len(data.speed)
    traj = np.tile(np.arange(N), len(grid))
    tau = np.repeat([g[0] for g in grid], N)
    stay = np.repeat([g[1] / 1000 for g in grid], N)
    pol = lstm_policy_factory(cfg, net, data)(traj, tau, stay)
    res = simulate(data, traj, pol, cfg["failures"], cfg["deployment"])
    rows = []
    for k, (ta, st) in enumerate(grid):
        r = rates(res, slice(k * N, (k + 1) * N))
        rows.append({"tau": ta, "min_stay_ms": st, **r, "J": objective(r, cfg["objective"])})
    df = pd.DataFrame(rows)
    b = df.loc[df["J"].idxmin()]
    return df, {"tau": float(b["tau"]), "min_stay_ms": float(b["min_stay_ms"])}


# ----------------------------------------------------------------------------- statistics
def per_traj_rates(res) -> pd.DataFrame:
    m = res["minutes"]
    return pd.DataFrame({"rlf_per_min": res["n_rlf"] / m, "pp_per_min": res["n_pp"] / m,
                         "ho_per_min": res["n_ho"] / m, "outage_frac": res["outage_frac"]})


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Mean of (b - a) over trajectories with a 95 % percentile CI (paired: same trajectories)."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ----------------------------------------------------------------------------- main
def main(stage: str, cfg: dict, out: Path, fast: bool = False) -> None:
    if fast:
        cfg = _fast(cfg)
    out.mkdir(parents=True, exist_ok=True)
    speeds = cfg["mobility"]["speeds_kmh"]
    # Trajectories (and Python TR 38.901 traces) always come from the Python producer, so the two
    # trace sources share identical UE paths; ns-3 only replaces the channel.
    py_dir = out / "traces"
    tr.build_all(cfg, py_dir, base_seed=1234)
    if stage == "export-trajectories":
        export_trajectories(cfg, py_dir, Path(cfg["ns3_trace_dir"]))
        return
    if cfg.get("trace_source", "python") == "ns3":
        trace_dir = out / "traces_ns3"
        _convert_ns3(cfg, trace_dir)
    else:
        trace_dir = py_dir
    if stage == "traces":
        return

    sims = {(s, sp): load_sim(cfg, trace_dir, s, sp) for s in tr.SPLITS for sp in speeds}
    trace_diagnostics(cfg, sims).to_csv(out / "trace_diagnostics.csv", index=False)
    # ---- A3 tuning on validation, per speed
    a3_rows, a3_best = [], {}
    for sp in speeds:
        df, best = tune_a3(cfg, sims[("val", sp)])
        a3_rows.append(df.assign(speed_kmh=sp))
        a3_best[sp] = best
        log.info("A3 %3d km/h: best hys %.0f dB, TTT %.0f ms", sp, best["hysteresis_db"], best["ttt_ms"])
    pd.concat(a3_rows).to_csv(out / "a3_grid_val.csv", index=False)
    (out / "a3_best.json").write_text(json.dumps({str(k): v for k, v in a3_best.items()}, indent=2))
    if stage == "tune":
        return

    lc = cfg["lstm"]
    stride = 5
    test_rows, lstm_grid_rows, clf_rows, boot_rows, chosen = [], [], [], [], {"a3": a3_best, "lstm": {}}

    def a3_fn(sp):
        return lambda B: a3_policy(cfg, a3_best[sp], B, sims[("val", sp)].dt_s)

    # A3 rollouts provide the initial state distribution (train) and the validation states
    def states(split, policy_for_speed):
        xs, ys, ss = [], [], []
        for sp in speeds:
            d = sims[(split, sp)]
            t, tj, sv = collect(cfg, d, policy_for_speed(sp), stride)
            x, y, s = make_xy(cfg, d, t, tj, sv)
            xs.append(x); ys.append(y); ss.append(s)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)

    x_tr, y_tr, s_tr = states("train", a3_fn)
    x_va, y_va, s_va = states("val", a3_fn)
    x_te, y_te, s_te = states("test", a3_fn)
    log.info("LSTM states: train %d (pos %.1f%%), val %d, test %d", len(y_tr), 100 * y_tr.mean(),
             len(y_va), len(y_te))

    a3_test = {}
    for sp in speeds:
        d = sims[("test", sp)]
        N = len(d.speed)
        res = simulate(d, np.arange(N), a3_policy(cfg, a3_best[sp], N, d.dt_s), cfg["failures"], cfg["deployment"])
        a3_test[sp] = res
        test_rows.append({"policy": "A3 (tuned per speed)", "speed_kmh": sp, "seed": -1, **rates(res)})

    for seed in cfg["seeds"]:
        net, hist = train_net(cfg, x_tr, y_tr, s_tr, x_va, y_va, s_va, seed)
        X, Y, S = x_tr, y_tr, s_tr
        for it in range(lc["dagger_iters"]):
            def lstm_fn(sp, net=net):
                d = sims[("train", sp)]
                mk = lstm_policy_factory(cfg, net, d)
                return lambda B: mk(np.arange(B), np.full(B, 0.5), np.zeros(B))
            xd, yd, sd = states("train", lstm_fn)
            X, Y, S = np.concatenate([X, xd]), np.concatenate([Y, yd]), np.concatenate([S, sd])
            net, h2 = train_net(cfg, X, Y, S, x_va, y_va, s_va, seed)
            hist += [{**h, "dagger_iter": it + 1} for h in h2]
        pd.DataFrame(hist).to_csv(out / f"lstm_history_seed{seed}.csv", index=False)
        clf_rows.append({"seed": seed, **classification_metrics(net, x_te, y_te, s_te)})
        import torch
        torch.save(net.state_dict(), out / f"lstm_seed{seed}.pt")
        for sp in speeds:
            gdf, best = tune_lstm(cfg, net, sims[("val", sp)])
            lstm_grid_rows.append(gdf.assign(speed_kmh=sp, seed=seed))
            chosen["lstm"][f"{sp}/seed{seed}"] = best
            d = sims[("test", sp)]
            N = len(d.speed)
            pol = lstm_policy_factory(cfg, net, d)(np.arange(N), np.full(N, best["tau"]),
                                                   np.full(N, best["min_stay_ms"] / 1000))
            res = simulate(d, np.arange(N), pol, cfg["failures"], cfg["deployment"])
            test_rows.append({"policy": "LSTM predictor", "speed_kmh": sp, "seed": seed, **rates(res)})
            pa, pb = per_traj_rates(a3_test[sp]), per_traj_rates(res)
            for met in ("rlf_per_min", "pp_per_min", "ho_per_min", "outage_frac"):
                m, lo, hi = paired_bootstrap(pa[met].to_numpy(), pb[met].to_numpy(), seed=seed)
                boot_rows.append({"speed_kmh": sp, "seed": seed, "metric": met,
                                  "lstm_minus_a3": m, "ci95_lo": lo, "ci95_hi": hi})
            log.info("%3d km/h seed %d: RLF/min A3 %.3f vs LSTM %.3f | PP/min %.3f vs %.3f",
                     sp, seed, test_rows[speeds.index(sp)]["rlf_per_min"], test_rows[-1]["rlf_per_min"],
                     test_rows[speeds.index(sp)]["pp_per_min"], test_rows[-1]["pp_per_min"])

    pd.DataFrame(test_rows).to_csv(out / "test_metrics.csv", index=False)
    pd.concat(lstm_grid_rows).to_csv(out / "lstm_grid_val.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(out / "lstm_classification_test.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "paired_bootstrap.csv", index=False)
    (out / "chosen_params.json").write_text(json.dumps(chosen, indent=2, default=str))
    freeze_manifest(cfg, out, "handover", extra={"fast": fast,
                                                 "trace_source": cfg.get("trace_source", "python")})


def trace_diagnostics(cfg, sims) -> pd.DataFrame:
    """Sanity numbers for the report's simulation section: an oracle that always serves the
    strongest cell bounds what any policy can achieve; large per-sample RSRP steps would mean
    unavoidable failures (the reason the soft LOS transition exists)."""
    from testbed.handover.channel import noise_per_re_dbm
    n = 10 ** (noise_per_re_dbm(cfg["deployment"]) / 10)
    rows = []
    for (split, sp), d in sims.items():
        lin = 10 ** (d.rsrp / 10)
        s = lin.max(-1)
        oracle = 10 * np.log10(s / (lin.sum(-1) - s + n))
        step = np.abs(np.diff(d.rsrp, axis=1))
        rows.append({"split": split, "speed_kmh": sp, "n_traj": len(d.speed),
                     "oracle_sinr_p1_db": float(np.percentile(oracle, 1)),
                     "oracle_sinr_p50_db": float(np.percentile(oracle, 50)),
                     "oracle_frac_below_qout": float((oracle < cfg["failures"]["q_out_db"]).mean()),
                     "rsrp_step_p99_db": float(np.percentile(step, 99)),
                     "rsrp_step_max_db": float(step.max())})
    return pd.DataFrame(rows)


def export_trajectories(cfg, py_dir: Path, out: Path) -> None:
    """UE paths as CSV for ns3/nr-rsrp-traces.cc (traj,time_s,speed_kmh,x,y)."""
    out.mkdir(parents=True, exist_ok=True)
    for split in tr.SPLITS:
        for sp in cfg["mobility"]["speeds_kmh"]:
            z = tr.load(py_dir / f"{split}_{int(sp)}kmh.npz")
            N, T, _ = z["pos"].shape
            df = pd.DataFrame({"traj": np.repeat(np.arange(N), T), "time_s": np.tile(z["t_s"], N),
                               "speed_kmh": np.repeat(z["speed"], T),
                               "x": z["pos"][..., 0].ravel(), "y": z["pos"][..., 1].ravel()})
            df.to_csv(out / f"{split}_{int(sp)}kmh_traj.csv", index=False, float_format="%.3f")
    log.info("trajectories written to %s", out)


def _convert_ns3(cfg, trace_dir: Path) -> None:
    src = Path(cfg["ns3_trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    for split in tr.SPLITS:
        for sp in cfg["mobility"]["speeds_kmh"]:
            dst = trace_dir / f"{split}_{int(sp)}kmh.npz"
            if dst.exists():
                continue
            csv = src / f"{split}_{int(sp)}kmh.csv"
            if not csv.exists():
                raise FileNotFoundError(f"{csv}: run ns3/run_all.sh first")
            np.savez_compressed(dst, **tr.from_ns3_csv(csv, cfg["channel"]["meas_noise_db"],
                                                       seed=hash((split, sp)) % 2**31))
