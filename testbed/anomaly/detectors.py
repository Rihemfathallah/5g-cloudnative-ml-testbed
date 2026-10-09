"""Detector library. All models share one interface so the evaluator never changes:

    fit(train, val)   unsupervised models see NORMAL training windows only ("clean training"
                      assumption, stated in the report); the RF reference sees labels
    score(ws)         higher = more anomalous
    threshold         chosen on VALIDATION data by a fixed, stated rule -- never on test
    explain(ws)       optional per-column attribution
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from testbed.data.features import WindowSet

log = logging.getLogger(__name__)


class Detector(ABC):
    name: str = "base"

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.threshold: float = np.inf

    @abstractmethod
    def fit(self, train: WindowSet, val: WindowSet) -> "Detector": ...

    @abstractmethod
    def score(self, ws: WindowSet) -> np.ndarray: ...

    def predict(self, ws: WindowSet) -> np.ndarray:
        return self.score(ws) > self.threshold

    def explain(self, ws: WindowSet) -> pd.DataFrame | None:
        return None


def _normal(ws: WindowSet) -> np.ndarray:
    return (ws.meta["label"] == 0).to_numpy()


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising window F1 on labelled (0/1) validation windows."""
    m = labels >= 0
    s, y = scores[m], labels[m]
    if y.sum() == 0:
        return float(np.quantile(s, 0.99))
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    # only cut between distinct score values
    distinct = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    f1 = np.where(distinct, f1, -1)
    i = int(np.argmax(f1))
    nxt = s_sorted[i + 1] if i + 1 < len(s_sorted) else s_sorted[i] - 1.0
    return float((s_sorted[i] + nxt) / 2)   # predict = score > thr includes s_sorted[i]


def fa_budget_threshold(scores, meta, budget_per_h, stride_s, merge_s) -> float:
    from testbed.anomaly.evaluate import false_alarms_per_hour
    cands = np.unique(np.quantile(scores[meta["label"].to_numpy() == 0], np.linspace(0.9, 1.0, 201)))
    for thr in cands:
        fa, _ = false_alarms_per_hour(scores > thr, meta, stride_s, merge_s)
        if fa <= budget_per_h:
            return float(thr)
    return float(cands[-1])


# ------------------------------------------------------------------------------- z-score
class ZScore(Detector):
    """Floor baseline: max |z| over all tabular features, z from normal training windows."""
    name = "zscore"

    def fit(self, train, val):
        Xn = train.X[_normal(train)]
        self.mu = Xn.mean(axis=0)
        sd = Xn.std(axis=0)
        self.sd = np.where(sd > 1e-9, sd, np.inf)   # constant features never alarm
        self.threshold = float(self.cfg.get("threshold", 3.0))
        return self

    def score(self, ws):
        return np.max(np.abs((ws.X - self.mu) / self.sd), axis=1)


# ------------------------------------------------------------------------------- Isolation Forest
class IForest(Detector):
    name = "iforest"

    def fit(self, train, val):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
        Xn = train.X[_normal(train)]
        self.scaler = RobustScaler().fit(Xn)
        self.model = IsolationForest(n_estimators=self.cfg.get("n_estimators", 300),
                                     max_samples=min(self.cfg.get("max_samples", 256), len(Xn)),
                                     random_state=self.seed, n_jobs=-1).fit(self.scaler.transform(Xn))
        s_val = self.score(val)
        if self.cfg.get("threshold_rule", "val_f1") == "fa_budget":
            self.threshold = fa_budget_threshold(s_val, val.meta, self.cfg["fa_budget_per_h"],
                                                 self.cfg["_stride_s"], self.cfg["_merge_s"])
        else:
            self.threshold = best_f1_threshold(s_val, val.meta["label"].to_numpy())
        return self

    def score(self, ws):
        return -self.model.score_samples(self.scaler.transform(ws.X))


# ------------------------------------------------------------------------------- LSTM autoencoder
class _LSTMAE:
    """Encoder LSTM -> latent -> repeated -> decoder LSTM -> per-step reconstruction."""

    def __new__(cls, n_in, hidden, latent, layers):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.LSTM(n_in, hidden, layers, batch_first=True)
                self.to_lat = nn.Linear(hidden, latent)
                self.from_lat = nn.Linear(latent, hidden)
                self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
                self.out = nn.Linear(hidden, n_in)

            def forward(self, x):
                _, (h, _) = self.enc(x)
                z = self.to_lat(h[-1])
                rep = self.from_lat(z).unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = self.dec(rep)
                return self.out(y)
        return Net()


class LSTMAutoencoder(Detector):
    name = "lstm_ae"

    def fit(self, train, val):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        c = self.cfg
        Xtr = torch.tensor(train.seq[_normal(train)])
        Xva = torch.tensor(val.seq[_normal(val)])
        self.net = _LSTMAE(Xtr.shape[2], c.get("hidden", 64), c.get("latent", 16), c.get("layers", 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=c.get("lr", 1e-3))
        loss_fn = torch.nn.MSELoss()
        bs, best, best_state, bad = c.get("batch_size", 128), np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        self.history = []
        for ep in range(c.get("epochs", 40)):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            tot = 0.0
            for i in range(0, len(Xtr), bs):
                xb = Xtr[perm[i:i + bs]]
                opt.zero_grad()
                loss = loss_fn(self.net(xb), xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(xb)
            vl = float(self._recon_err(Xva).mean())
            self.history.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": vl})
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c.get("patience", 6):
                    break
        self.net.load_state_dict(best_state)
        # Threshold: quantile of validation scores on NORMAL windows (does not use fault labels
        # beyond the normal/not-normal filter that also selects the training data).
        self.threshold = float(np.quantile(self.score(val)[_normal(val)],
                                           c.get("threshold_quantile", 0.99)))
        return self

    def _per_channel_err(self, X) -> np.ndarray:
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = X[i:i + 2048] if isinstance(X, torch.Tensor) else torch.tensor(X[i:i + 2048])
                out.append(((self.net(xb) - xb) ** 2).mean(dim=1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    def _recon_err(self, X):
        return self._per_channel_err(X).mean(axis=1)

    def score(self, ws):
        return self._recon_err(ws.seq)

    def explain(self, ws):
        err = self._per_channel_err(ws.seq)
        # mask channels are attributed to their underlying column
        base = [c.split("|missing")[0] for c in ws.seq_channels]
        df = pd.DataFrame(err, columns=ws.seq_channels).T.groupby(base).sum().T
        return df.div(df.sum(axis=1).replace(0, np.nan), axis=0)


# ------------------------------------------------------------------------------- supervised reference
class RandomForestRef(Detector):
    """Upper reference only (uses labels); never the headline result."""
    name = "rf_supervised"

    def fit(self, train, val):
        from sklearn.ensemble import RandomForestClassifier
        m = train.meta["label"].to_numpy() >= 0
        y = train.meta["label"].to_numpy()[m]
        if len(np.unique(y)) < 2:
            raise ValueError("RF reference needs both classes in training windows")
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.get("n_estimators", 300),
            min_samples_leaf=self.cfg.get("min_samples_leaf", 2),
            class_weight=self.cfg.get("class_weight", "balanced"),
            random_state=self.seed, n_jobs=-1).fit(train.X[m], y)
        self.threshold = 0.5
        return self

    def score(self, ws):
        return self.model.predict_proba(ws.X)[:, 1]


REGISTRY = {"zscore": ZScore, "iforest": IForest, "lstm_ae": LSTMAutoencoder, "rf": RandomForestRef}
DETERMINISTIC = {"zscore"}   # one seed is enough
