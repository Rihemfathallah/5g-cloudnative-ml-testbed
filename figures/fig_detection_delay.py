"""Figure: detection delay per fault type (box plot over detected test faults, all seeds) for one
model (default: lstm_ae). Reads results/<exp>/events.csv."""
import argparse
from pathlib import Path

import pandas as pd

from style import ENTITY, INK2, LABEL, check_not_synthetic, plt, save, setup, watermark

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("-o", "--out", default="report/figures")
ap.add_argument("--model", default="lstm_ae"); ap.add_argument("--allow-synthetic", action="store_true")
a = ap.parse_args()
res = Path(a.results)
check_not_synthetic(res, a.allow_synthetic)
setup()
ev = pd.read_csv(res / "events.csv")
ev = ev[(ev.model == a.model) & (~ev.excluded.astype(bool)) & (ev.detected == True)]  # noqa: E712
types = sorted(ev["type"].unique())
fig, ax = plt.subplots(figsize=(4.2, 0.3 * len(types) + 1.1))
data = [ev[ev["type"] == t]["delay_s"].to_numpy() for t in types]
bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                medianprops={"color": "white", "linewidth": 1.5},
                flierprops={"markersize": 3, "markerfacecolor": INK2, "markeredgecolor": "none"})
for b in bp["boxes"]:
    b.set(facecolor=ENTITY[a.model], edgecolor=ENTITY[a.model])
ax.set_yticks(range(1, len(types) + 1), [f"{t}  (n={len(d)})" for t, d in zip(types, data)])
ax.set_xlabel("delay: first alarm window end - t_applied (s)")
ax.set_title(f"{LABEL[a.model]}: detection delay", loc="left")
ax.grid(axis="y", visible=False)
watermark(fig, res)
save(fig, Path(a.out), f"detection_delay_{a.model}")
