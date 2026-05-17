"""
outputs/figures/plot_signal_distribution.py
Reproduces the two-panel Monthly LLM Signal figure from signals_clean.csv.

Left panel:  time series with president-coloured shaded regions
Right panel: violin + strip plot with mean annotations

Reads:  data/processed/signals_clean.csv
Writes: outputs/figures/slides_signal_distribution.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, "..", ".."))
SIGNAL_CSV = os.path.join(_ROOT, "data", "processed", "signals_clean.csv")
OUT_PATH   = os.path.join(_HERE, "slides_signal_distribution.png")

# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "Macri": "#1565C0",   # dark blue
    "AF":    "#1B5E20",   # dark green
    "Milei": "#BF360C",   # dark orange
}
BG_COLORS = {
    "Macri": "#DDEEFF",   # pale blue  — time series shading only
    "AF":    "#DDEEDD",   # pale green — time series shading only
    "Milei": "#FFE4CC",   # warm peach — time series shading only
}
VIOLIN_COLORS = {
    "Macri": "#6BAED6",   # medium steel blue
    "AF":    "#43A047",   # medium forest green
    "Milei": "#F4702A",   # warm amber orange
}
PRES_ORDER    = ["Macri", "AF", "Milei"]
PRES_LABELS   = {"Macri": "Macri", "AF": "AF", "Milei": "Milei"}
VIOLIN_LABELS = {"Macri": "Macri\n(n=49)", "AF": "AF\n(n=47)", "Milei": "Milei\n(n=29)"}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(SIGNAL_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7.5))
gs  = GridSpec(1, 2, width_ratios=[2.6, 1], wspace=0.06)
ax_ts  = fig.add_subplot(gs[0])
ax_vio = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — time series
# ═══════════════════════════════════════════════════════════════════════════════

for pres in PRES_ORDER:
    sub = df[df["president"] == pres].sort_values("date")
    # Shaded background region
    ax_ts.axvspan(sub["date"].iloc[0], sub["date"].iloc[-1],
                  color=BG_COLORS[pres], alpha=0.55, zorder=0)
    # Fill under/over the line
    ax_ts.fill_between(sub["date"], sub["signal_main"], 0,
                       color=VIOLIN_COLORS[pres], alpha=0.25, zorder=1)
    # Line
    ax_ts.plot(sub["date"], sub["signal_main"],
               color=COLORS[pres], linewidth=1.8, zorder=2)
    # President label at top of region
    mid_date = sub["date"].iloc[len(sub)//2]
    ax_ts.text(mid_date, 1.75, pres,
               ha="center", va="top", fontsize=14, fontweight="bold",
               color=COLORS[pres], zorder=3)

# Zero line
ax_ts.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

# Davos annotation
davos_date = pd.Timestamp("2024-01-01")
davos_val  = df.loc[df["date"] == davos_date, "signal_main"].values[0]
ax_ts.annotate(
    f"Davos Jan 2024\n+{davos_val:.2f}z",
    xy=(davos_date, davos_val),
    xytext=(pd.Timestamp("2022-06-01"), davos_val - 0.05),
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.8),
    zorder=5,
)

ax_ts.set_ylabel("Hawkishness z-score", fontsize=12)
ax_ts.set_xlabel("")
ax_ts.set_xlim(df["date"].min(), df["date"].max())
ax_ts.set_ylim(-1.85, 1.95)
ax_ts.set_title("Monthly LLM Signal", fontsize=15, fontweight="bold", pad=10)
ax_ts.tick_params(axis="both", labelsize=10)
ax_ts.spines[["top", "right"]].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — violin + strip
# ═══════════════════════════════════════════════════════════════════════════════

ax_vio.set_title("Distribution by President", fontsize=15, fontweight="bold", pad=10)
ax_vio.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

positions = {p: i for i, p in enumerate(PRES_ORDER)}
rng = np.random.default_rng(42)

for pres in PRES_ORDER:
    sub  = df[df["president"] == pres]["signal_main"].dropna().values
    pos  = positions[pres]
    col  = COLORS[pres]
    bcol = VIOLIN_COLORS[pres]

    # Violin
    parts = ax_vio.violinplot(sub, positions=[pos], widths=0.7,
                               showmeans=False, showmedians=False,
                               showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(bcol)
        pc.set_edgecolor(col)
        pc.set_alpha(0.88)
        pc.set_linewidth(1.2)

    # Jittered strip
    jitter = rng.uniform(-0.07, 0.07, size=len(sub))
    ax_vio.scatter(pos + jitter, sub,
                   color=col, s=28, alpha=0.7, zorder=3, linewidths=0)

    # Mean line
    mean_val = sub.mean()
    ax_vio.plot([pos - 0.18, pos + 0.18], [mean_val, mean_val],
                color="black", linewidth=2.0, zorder=4)

    # Mean annotation at bottom
    ax_vio.text(pos, -1.75,
                f"{mean_val:+.2f}z",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=col, zorder=5)

ax_vio.set_xticks(list(positions.values()))
ax_vio.set_xticklabels(
    [f"{p}\n(n={len(df[df['president']==p])})" for p in PRES_ORDER],
    fontsize=11
)
ax_vio.set_ylim(-1.95, 1.95)
ax_vio.set_ylabel("")
ax_vio.tick_params(axis="y", labelsize=10)
ax_vio.spines[["top", "right"]].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_PATH}")
