"""Figure: A3 tuning objective J over hysteresis x TTT, one panel per speed (validation), with the
chosen setting marked. Shows whether the optimum sits on the grid edge (then the grid must grow).
Reads results/handover/a3_grid_val.csv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import INK, plt, save, setup

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results/handover"); ap.add_argument("-o", "--out", default="report/figures")
a = ap.parse_args()
setup()
g = pd.read_csv(Path(a.results) / "a3_grid_val.csv")
speeds = sorted(g.speed_kmh.unique())
fig, axes = plt.subplots(1, len(speeds), figsize=(1.9 * len(speeds), 2.2), sharey=True)
for ax, sp in zip(np.atleast_1d(axes), speeds):
    p = g[g.speed_kmh == sp].pivot(index="hysteresis_db", columns="ttt_ms", values="J")
    # colour = J relative to the best setting of this speed, clipped at 3x: one terrible corner
    # (0 dB, 40 ms) would otherwise flatten every other cell to the same shade
    rel = np.clip(p.to_numpy() / np.nanmin(p.to_numpy()), 1, 3)
    ax.imshow(rel, cmap="Blues_r", vmin=1, vmax=3, aspect="auto", origin="lower")
    i, j = np.unravel_index(np.nanargmin(p.to_numpy()), p.shape)
    ax.scatter([j], [i], marker="o", s=40, facecolor="none", edgecolor=INK, linewidth=1.5)
    ax.set_xticks(range(p.shape[1]), [int(c) for c in p.columns], fontsize=6.5)
    ax.set_yticks(range(p.shape[0]), [f"{v:g}" for v in p.index])
    ax.set_xlabel("TTT (ms)")
    ax.set_title(f"{int(sp)} km/h", loc="left", fontsize=8)
    ax.grid(False)
np.atleast_1d(axes)[0].set_ylabel("hysteresis (dB)")
fig.suptitle("A3 tuning objective J / J_min on validation (darker = better, clipped at 3x; circle = chosen)", x=0.01, y=1.06, ha="left", fontsize=8)
save(fig, Path(a.out), "a3_grid")
