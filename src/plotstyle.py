"""Shared matplotlib style so every figure in the project looks the same."""
import matplotlib as mpl

# Categorical colours, always used in this fixed order (colour-blind checked).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NO_COLOR = "#a3a29c"   # neutral grey for the majority "no" class / reference lines
YES_COLOR = SERIES[0]  # blue for the "yes" class
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
