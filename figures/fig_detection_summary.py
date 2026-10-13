"""Figure: event recall, false alarms per hour and median detection delay per model (mean +- std
over seeds). Reads results/<exp>/summary.csv.   python3 figures/fig_detection_summary.py results/<exp>"""
import argparse
from pathlib import Path

import pandas as pd

from style import ENTITY, HATCH, INK2, SHORT, check_not_synthetic, plt, save, setup, watermark

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("-o", "--out", default="report/figures")
ap.add_argument("--allow-synthetic", action="store_true")
a = ap.parse_args()
res = Path(a.results)
check_not_synthetic(res, a.allow_synthetic)
setup()
s = pd.read_csv(res / "summary.csv")
order = [m for m in ("zscore", "iforest", "lstm_ae", "rf_supervised") if m in set(s.model)]
s = s.set_index("model").loc[order]
panels = [("event_recall", "Event recall", "{:.2f}"), ("false_alarms_per_h", "False alarms / h", "{:.1f}"),
          ("delay_median_s", "Median detection delay (s)", "{:.0f}")]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
for ax, (col, title, fmt) in zip(axes, panels):
    y, e = s[f"{col}_mean"], s[f"{col}_std"].fillna(0)
    bars = ax.bar(range(len(order)), y, yerr=e, width=0.6, color=[ENTITY[m] for m in order],
                  hatch=[HATCH.get(m, "") for m in order], edgecolor="white", linewidth=0,
                  error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    for b, v in zip(bars, y):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xticks(range(len(order)), [SHORT[m] for m in order], fontsize=7)
    ax.grid(axis="x", visible=False)
fig.suptitle(f"Test-split detection, {int(s['n_seeds'].max())} seeds (error bars: std)", x=0.01, y=1.06, ha="left", fontsize=9)
watermark(fig, res)
save(fig, Path(a.out), "detection_summary")
