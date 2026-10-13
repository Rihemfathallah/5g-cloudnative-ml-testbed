"""Figure: which metrics explain LSTM-AE alarms, per fault type (share of reconstruction error,
top-k columns per type). Reads results/<exp>/attribution.csv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from style import INK, check_not_synthetic, plt, save, setup, watermark

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("-o", "--out", default="report/figures")
ap.add_argument("--top", type=int, default=3); ap.add_argument("--allow-synthetic", action="store_true")
a = ap.parse_args()
res = Path(a.results)
check_not_synthetic(res, a.allow_synthetic)
setup()
at = pd.read_csv(res / "attribution.csv").groupby(["fault_type", "column"])["share"].mean().reset_index()
types = sorted(at.fault_type.unique())
cols = sorted({c for t in types for c in at[at.fault_type == t].nlargest(a.top, "share")["column"]})
M = np.zeros((len(types), len(cols)))
for i, t in enumerate(types):
    sub = at[at.fault_type == t].set_index("column")["share"]
    M[i] = [sub.get(c, 0.0) for c in cols]
fig, ax = plt.subplots(figsize=(0.45 * len(cols) + 1.6, 0.35 * len(types) + 1.4))
ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.01, M.max()), aspect="auto")
for i in range(len(types)):
    for j in range(len(cols)):
        if M[i, j] >= 0.05:
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 * M.max() else INK)
ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=6.5)
ax.set_yticks(range(len(types)), types)
ax.grid(False)
ax.set_title(f"LSTM-AE: share of reconstruction error (top {a.top} columns per fault type)", loc="left")
watermark(fig, res)
save(fig, Path(a.out), "attribution")
