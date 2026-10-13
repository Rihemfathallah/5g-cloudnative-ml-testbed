"""Figure: RLF, ping-pong and handover rates per UE speed, tuned A3 vs LSTM predictor
(LSTM: mean +- std over seeds). Reads results/handover/test_metrics.csv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import ENTITY, INK2, plt, save, setup

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results/handover"); ap.add_argument("-o", "--out", default="report/figures")
a = ap.parse_args()
setup()
d = pd.read_csv(Path(a.results) / "test_metrics.csv")
speeds = sorted(d.speed_kmh.unique())
pols = ["A3 (tuned per speed)", "LSTM predictor"]
panels = [("rlf_per_min", "Radio link failures / UE / min"), ("pp_per_min", "Ping-pongs / UE / min"),
          ("ho_per_min", "Handovers / UE / min")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
x = np.arange(len(speeds))
for ax, (col, title) in zip(axes, panels):
    for k, p in enumerate(pols):
        g = d[d.policy == p].groupby("speed_kmh")[col]
        m, s = g.mean().reindex(speeds), g.std(ddof=0).reindex(speeds).fillna(0)
        ax.bar(x + (k - 0.5) * 0.36, m, 0.34, yerr=s, color=ENTITY[p], label=p, linewidth=0,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, [f"{int(v)}" for v in speeds])
    ax.set_xlabel("UE speed (km/h)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="upper left", fontsize=7)
fig.suptitle("Held-out test trajectories; both policies tuned per speed on validation with the same objective",
             x=0.01, y=1.06, ha="left", fontsize=8)
save(fig, Path(a.out), "handover_rates")
