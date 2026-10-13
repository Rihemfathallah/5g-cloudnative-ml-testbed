"""Figure: event recall per fault type and severity, one panel per model (heatmap, mean over seeds,
n faults printed in each cell). Reads results/<exp>/event_summary.csv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import INK, LABEL, check_not_synthetic, plt, save, setup, watermark

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("-o", "--out", default="report/figures")
ap.add_argument("--allow-synthetic", action="store_true")
a = ap.parse_args()
res = Path(a.results)
check_not_synthetic(res, a.allow_synthetic)
setup()
e = pd.read_csv(res / "event_summary.csv")
models = [m for m in ("zscore", "iforest", "lstm_ae", "rf_supervised") if m in set(e.model)]
types = sorted(e["type"].unique())
sev = ["mild", "strong"]
fig, axes = plt.subplots(1, len(models), figsize=(1.8 * len(models) + 0.8, 0.35 * len(types) + 1.2), sharey=True)
axes = np.atleast_1d(axes)
cmap = plt.get_cmap("Blues")    # sequential: one hue, light -> dark
for ax, m in zip(axes, models):
    g = e[e.model == m].groupby(["type", "severity"]).agg(r=("event_recall", "mean"), n=("n_faults", "first"))
    M = np.full((len(types), 2), np.nan)
    N = np.zeros((len(types), 2), int)
    for (t, s_), row in g.iterrows():
        if s_ in sev:
            M[types.index(t), sev.index(s_)] = row.r
            N[types.index(t), sev.index(s_)] = row.n
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(types)):
        for j in range(2):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}\nn={N[i, j]}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks([0, 1], sev)
    ax.set_yticks(range(len(types)), types)
    ax.set_title(LABEL[m].split(" (")[0], loc="left", fontsize=8)
    ax.grid(False)
fig.suptitle("Event recall by fault type and severity (test split)", x=0.01, y=1.06, ha="left", fontsize=9)
watermark(fig, res)
save(fig, Path(a.out), "detection_by_type")
