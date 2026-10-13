"""Shared figure style. Palette = validated reference categorical slots 1-4 (scripts/validate_palette.js:
all checks pass on light; slots 3-4 are below 3:1 contrast, so every figure also carries a legend
and/or direct labels). Colour follows the ENTITY, never its rank: a model keeps its colour in
every figure, whichever subset is plotted."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ENTITY = {"iforest": SLOTS[0], "lstm_ae": SLOTS[1], "zscore": SLOTS[2], "rf_supervised": SLOTS[3],
          "A3 (tuned per speed)": SLOTS[0], "LSTM predictor": SLOTS[1]}
LABEL = {"iforest": "Isolation Forest", "lstm_ae": "LSTM autoencoder", "zscore": "z-score baseline",
         "rf_supervised": "Random Forest (supervised ref.)"}
SHORT = {"iforest": "IForest", "lstm_ae": "LSTM-AE", "zscore": "z-score", "rf_supervised": "RF ref."}
HATCH = {"rf_supervised": "//"}      # reference model is also marked by texture, not colour alone


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


def check_not_synthetic(results: Path, allow: bool) -> None:
    """Refuse to draw report figures from the synthetic stand-in."""
    if (results / "README_SYNTHETIC.txt").exists() and not allow:
        sys.exit(f"{results} holds SYNTHETIC smoke results; they never go in the report "
                 "(pass --allow-synthetic only to test the figure code)")


def watermark(fig, results: Path) -> None:
    if (results / "README_SYNTHETIC.txt").exists():
        fig.text(0.5, 0.5, "SYNTHETIC - NOT A RESULT", ha="center", va="center", fontsize=22,
                 color="#d03b3b", alpha=0.25, rotation=20)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(out / f"{name}.pdf")
