"""
signal/compile_signals.py — Compile all three fiscal hawkishness signals
─────────────────────────────────────────────────────────────────────────
Merges the zero-shot LLM signal, the few-shot LLM signal, and the
dictionary signal into a single clean monthly CSV suitable for the DFM /
Local Projections pipeline.

Construction rules
──────────────────
1. Transition months (2019-12, 2023-12): keep INCOMING president only.
     2019-12 → AF (inaugurated Dec 10, 2019)
     2023-12 → Milei (inaugurated Dec 10, 2023)
2. Months with zero fiscal paragraphs (AF 2020-01, AF 2023-11): fill
   all signal columns with 0 and flag as `is_zero_fill=True`.
   Rationale: absence of fiscal communication is treated as neutral
   stance, not imputed from adjacent months (CLAUDE.md Section 3.1).
3. Thin month (Milei 2026-02, P_t=1 — ceremonial speech): retained
   with `is_thin=True` flag so the econometrics team can drop if needed.

Outputs
───────
    data/processed/signals_clean.csv           monthly panel (primary)
    outputs/figures/signals_timeseries.png     main time-series figure
    outputs/figures/signals_scatter.png        ZS vs FS vs Dict scatters
    outputs/figures/signals_by_president.png   mean ± std bar chart

Usage
──────
    cd ~/Desktop/Masters-Project
    source .venv/bin/activate
    python signal/compile_signals.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.abspath(os.path.join(_HERE, ".."))
MONTHLY_LLM   = os.path.join(_ROOT, "data", "interim", "monthly_signal_llm.csv")
MONTHLY_FS    = os.path.join(_ROOT, "data", "interim", "monthly_signal_llm_fewshot.csv")
MONTHLY_DICT  = os.path.join(_ROOT, "data", "interim", "monthly_signal.csv")
OUT_CSV       = os.path.join(_ROOT, "data", "processed", "signals_clean.csv")
FIGURES_DIR   = os.path.join(_ROOT, "outputs", "figures")

# ── Config ────────────────────────────────────────────────────────────────────
PRES_ORDER  = ["Macri", "AF", "Milei"]
PRES_COLORS = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}
PRES_LABELS = {"Macri": "Macri (2015–19)", "AF": "Alberto F. (2019–23)", "Milei": "Milei (2023–)"}

# Inauguration dates — used for regime shading and vertical lines
INAUG = {
    "Macri": pd.Timestamp("2015-12-10"),
    "AF":    pd.Timestamp("2019-12-10"),
    "Milei": pd.Timestamp("2023-12-10"),
}

# Transition months: incoming president takes the whole month
TRANSITION = {
    "2019-12": "AF",    # AF inaugurated Dec 10, 2019
    "2023-12": "Milei", # Milei inaugurated Dec 10, 2023
}

# Thin-month threshold (flag but retain)
THIN_THRESHOLD = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_resolve(path: str, signal_col: str, rename_to: str) -> pd.DataFrame:
    """
    Load a monthly signal file and apply the transition-month rule.

    Returns a DataFrame with columns: year_month, president, <rename_to>,
    n_fiscal_paras (from source).
    """
    df = pd.read_csv(path)
    df = df[df["president"].isin(PRES_ORDER)].copy()

    out_rows = []
    for ym, grp in df.groupby("year_month"):
        if ym in TRANSITION:
            # Keep incoming president only
            incoming = TRANSITION[ym]
            row = grp[grp["president"] == incoming]
            if row.empty:
                # Incoming president had no speeches that month — skip (zero-fill later)
                continue
            out_rows.append(row.iloc[0])
        else:
            # Should be only one president for this month
            if len(grp) > 1:
                print(f"  WARNING: multiple presidents in {ym} outside transition months — keeping first")
            out_rows.append(grp.iloc[0])

    resolved = pd.DataFrame(out_rows).reset_index(drop=True)
    resolved = resolved.rename(columns={signal_col: rename_to})
    cols = ["year_month", "president", rename_to, "n_fiscal_paras"]
    return resolved[[c for c in cols if c in resolved.columns]].copy()


def build_clean_panel(llm: pd.DataFrame, fs: pd.DataFrame,
                      d: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three signals into a single panel, zero-fill missing months,
    and add quality flags.
    """
    # Merge on year_month (president should be consistent after transition resolution)
    panel = llm.merge(
        fs.rename(columns={"n_fiscal_paras": "n_fiscal_paras_fs"}),
        on=["year_month", "president"], how="outer",
    ).merge(
        d.rename(columns={"n_fiscal_paras": "n_fiscal_paras_dict"}),
        on=["year_month", "president"], how="outer",
    )

    # Generate complete monthly date range
    date_min = pd.to_datetime(panel["year_month"]).min()
    date_max = pd.to_datetime(panel["year_month"]).max()
    full_range = pd.date_range(date_min, date_max, freq="MS")
    full_ym    = full_range.strftime("%Y-%m").tolist()

    # Determine president assignment for every month in the full range
    def month_president(ym: str) -> str:
        if ym in TRANSITION:
            return TRANSITION[ym]
        dt = pd.Timestamp(ym)
        if dt < INAUG["AF"]:
            return "Macri"
        elif dt < INAUG["Milei"]:
            return "AF"
        else:
            return "Milei"

    full_df = pd.DataFrame({
        "year_month": full_ym,
        "president":  [month_president(ym) for ym in full_ym],
    })

    # Left-join panel onto the full date range
    merged = full_df.merge(panel, on=["year_month", "president"], how="left")

    # Zero-fill missing months and add flags
    signal_cols = ["signal_main", "signal_robust", "signal_dictionary"]
    missing_mask = merged["signal_main"].isna()
    merged.loc[missing_mask, signal_cols] = 0.0
    merged["n_fiscal_paras"] = merged["n_fiscal_paras"].fillna(0).astype(int)

    merged["is_zero_fill"] = missing_mask
    merged["is_thin"]      = (merged["n_fiscal_paras"] > 0) & \
                             (merged["n_fiscal_paras"] <= THIN_THRESHOLD)

    # Datetime column for plotting
    merged["date"] = pd.to_datetime(merged["year_month"])

    return merged.sort_values("date").reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def shade_regimes(ax, alpha: float = 0.08):
    """Add background shading by presidential regime."""
    regime_spans = [
        ("Macri", pd.Timestamp("2015-12-01"), INAUG["AF"]),
        ("AF",    INAUG["AF"],                INAUG["Milei"]),
        ("Milei", INAUG["Milei"],             pd.Timestamp("2026-06-01")),
    ]
    for pres, start, end in regime_spans:
        ax.axvspan(start, end, alpha=alpha, color=PRES_COLORS[pres], zorder=0)
    for _, ts in INAUG.items():
        ax.axvline(ts, color="grey", linewidth=0.9, linestyle=":", alpha=0.7, zorder=1)


def plot_timeseries(panel: pd.DataFrame):
    """
    Main figure: three-panel time series, one signal per row, with
    regime shading.  Signals plotted with their own colour per president
    so regime transitions are immediately visible.
    """
    signal_info = [
        ("signal_main",       "LLM Signal — Zero-shot  (signal_main)"),
        ("signal_robust",     "LLM Signal — Few-shot  (signal_robust)"),
        ("signal_dictionary", "Dictionary Signal  (signal_dictionary)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Argentine Presidential Fiscal Hawkishness Signals  (monthly z-scores)",
                 fontsize=13, y=0.98)

    for ax, (col, title) in zip(axes, signal_info):
        shade_regimes(ax)
        for pres in PRES_ORDER:
            sub = panel[panel["president"] == pres].copy()
            # Mark zero-fill and thin months
            normal = sub[~sub["is_zero_fill"] & ~sub["is_thin"]]
            zeros  = sub[sub["is_zero_fill"]]
            thin   = sub[sub["is_thin"]]
            ax.plot(normal["date"], normal[col],
                    color=PRES_COLORS[pres], linewidth=1.8, label=PRES_LABELS[pres])
            ax.plot(thin["date"], thin[col],
                    color=PRES_COLORS[pres], linewidth=1.8, linestyle="--")
            if not zeros.empty:
                ax.scatter(zeros["date"], zeros[col],
                           color=PRES_COLORS[pres], marker="x", s=60, zorder=5,
                           label=f"{pres} (zero-fill)")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_ylabel("Z-score", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", pad=4)
        ax.set_ylim(-2.2, 2.2)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Legend on top panel only
    handles = [mpatches.Patch(color=PRES_COLORS[p], label=PRES_LABELS[p])
               for p in PRES_ORDER]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.8)

    axes[-1].set_xlabel("Month", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIGURES_DIR, "signals_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_scatter(panel: pd.DataFrame):
    """
    2×1 scatter: (left) zero-shot vs few-shot, (right) zero-shot vs dictionary.
    Both cross-signal comparisons with Pearson r and Spearman ρ annotated.
    """
    valid = panel[~panel["is_zero_fill"]].dropna(
        subset=["signal_main", "signal_robust", "signal_dictionary"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Signal Cross-Validation Scatter", fontsize=12)

    comparisons = [
        ("signal_main",   "signal_robust",
         "signal_main (z-score)", "signal_robust (z-score)",
         "signal_main vs signal_robust"),
        ("signal_main",   "signal_dictionary",
         "signal_main (z-score)", "signal_dictionary (z-score)",
         "signal_main vs signal_dictionary"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, comparisons):
        for pres in PRES_ORDER:
            sub = valid[valid["president"] == pres]
            ax.scatter(sub[xcol], sub[ycol],
                       color=PRES_COLORS[pres], alpha=0.6, s=28, label=PRES_LABELS[pres],
                       zorder=3)
        # 45-degree reference line
        lo = valid[[xcol, ycol]].min().min() - 0.1
        hi = valid[[xcol, ycol]].max().max() + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.35, zorder=2)
        # Correlation stats
        r   = valid[xcol].corr(valid[ycol])
        rho = valid[xcol].corr(valid[ycol], method="spearman")
        ax.text(0.04, 0.95,
                f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(linewidth=0.4, alpha=0.4)

    handles = [mpatches.Patch(color=PRES_COLORS[p], label=PRES_LABELS[p])
               for p in PRES_ORDER]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "signals_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_by_president(panel: pd.DataFrame):
    """
    Bar chart: mean z-score ± 1 std by president, for all three signals.
    Makes the Milei–AF gap immediately visible across all methods.
    """
    valid = panel[~panel["is_zero_fill"]]
    signals = [
        ("signal_main",       "signal_main",       "#7B1FA2"),
        ("signal_robust",     "signal_robust",     "#AB47BC"),
        ("signal_dictionary", "signal_dictionary", "#607D8B"),
    ]

    n_sig = len(signals)
    n_pres = len(PRES_ORDER)
    width  = 0.22
    x      = np.arange(n_pres)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Signal Comparison by Presidential Regime", fontsize=12)

    # ── Left: mean ± std bar chart ────────────────────────────────────────────
    ax = axes[0]
    offsets = np.linspace(-(n_sig-1)/2 * width, (n_sig-1)/2 * width, n_sig)
    for (col, label, color), offset in zip(signals, offsets):
        means = [valid[valid["president"] == p][col].mean() for p in PRES_ORDER]
        stds  = [valid[valid["president"] == p][col].std()  for p in PRES_ORDER]
        bars  = ax.bar(x + offset, means, width, label=label,
                       color=color, alpha=0.85,
                       yerr=stds, capsize=3, error_kw={"linewidth": 1})
        for bar, m in zip(bars, means):
            ypos = bar.get_height() + (0.06 if m >= 0 else -0.18)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{m:+.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PRES_LABELS[p] for p in PRES_ORDER], fontsize=9)
    ax.set_ylabel("Mean z-score (± 1 std)", fontsize=9)
    ax.set_title("Mean signal by president (all months)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # ── Right: overlaid time series (all three signals, colour = signal type)
    ax = axes[1]
    line_styles = ["-", "--", ":"]
    for (col, label, color), ls in zip(signals, line_styles):
        for pres in PRES_ORDER:
            sub = valid[valid["president"] == pres].sort_values("date")
            ax.plot(sub["date"], sub[col], color=color, linewidth=1.5,
                    linestyle=ls, alpha=0.85)

    shade_regimes(ax, alpha=0.06)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylabel("Z-score", fontsize=9)
    ax.set_title("All three signals overlaid", fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    # Combined legend: signal types
    sig_handles = [
        plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=1.8, label=label)
        for (_, label, color), ls in zip(signals, line_styles)
    ]
    regime_handles = [
        mpatches.Patch(color=PRES_COLORS[p], alpha=0.25, label=PRES_LABELS[p])
        for p in PRES_ORDER
    ]
    ax.legend(handles=sig_handles + regime_handles, fontsize=7.5, loc="upper left",
              framealpha=0.85, ncol=2)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "signals_by_president.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Load and resolve each signal ──────────────────────────────────────
    for path, label in [(MONTHLY_LLM, "LLM zero-shot"),
                        (MONTHLY_FS,  "LLM few-shot"),
                        (MONTHLY_DICT,"Dictionary")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    print("Loading signals...")
    llm  = load_and_resolve(MONTHLY_LLM,  "net_hawkish_llm_z",  "signal_main")
    fs   = load_and_resolve(MONTHLY_FS,   "net_hawkish_llm_z",  "signal_robust")
    d    = load_and_resolve(MONTHLY_DICT, "net_hawkish_z",       "signal_dictionary")

    print(f"  LLM zero-shot : {len(llm)} month-rows after transition resolution")
    print(f"  LLM few-shot  : {len(fs)} month-rows after transition resolution")
    print(f"  Dictionary    : {len(d)} month-rows after transition resolution")

    # ── 2. Build clean panel ──────────────────────────────────────────────────
    print("\nBuilding clean panel...")
    panel = build_clean_panel(llm, fs, d)

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"  Total months in panel   : {len(panel)}")
    print(f"  Zero-fill months        : {panel['is_zero_fill'].sum()}")
    zf = panel[panel["is_zero_fill"]]
    for _, r in zf.iterrows():
        print(f"    {r['year_month']}  {r['president']}  [zero fiscal paragraphs]")
    print(f"  Thin months (P_t ≤ {THIN_THRESHOLD}) : {panel['is_thin'].sum()}")
    th = panel[panel["is_thin"]]
    for _, r in th.iterrows():
        print(f"    {r['year_month']}  {r['president']}  P_t={r['n_fiscal_paras']}")

    print("\n── Signal means by president ────────────────────────────────────")
    valid = panel[~panel["is_zero_fill"]]
    header = f"  {'President':<8}  {'N':>4}  {'signal_main':>12}  {'signal_robust':>14}  {'signal_dict':>12}"
    print(header)
    for pres in PRES_ORDER:
        sub = valid[valid["president"] == pres]
        zs  = sub["signal_main"].mean()
        fss = sub["signal_robust"].mean()
        dct = sub["signal_dictionary"].mean()
        print(f"  {pres:<8}  {len(sub):>4}  {zs:>+12.3f}  {fss:>+14.3f}  {dct:>+12.3f}")

    print("\n── Cross-signal correlations (excluding zero-fill) ──────────────")
    for (a, b, label) in [
        ("signal_main",   "signal_robust",     "signal_main vs signal_robust    "),
        ("signal_main",   "signal_dictionary", "signal_main vs signal_dictionary"),
        ("signal_robust", "signal_dictionary", "signal_robust vs signal_dict    "),
    ]:
        sub = valid.dropna(subset=[a, b])
        r   = sub[a].corr(sub[b])
        rho = sub[a].corr(sub[b], method="spearman")
        print(f"  {label}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")

    # ── 4. Save CSV ───────────────────────────────────────────────────────────
    out_cols = [
        "year_month", "president", "date",
        "signal_main",       # primary signal (zero-shot LLM)
        "signal_robust",     # few-shot robustness
        "signal_dictionary", # dictionary robustness
        "n_fiscal_paras",
        "is_zero_fill",
        "is_thin",
    ]
    panel[[c for c in out_cols if c in panel.columns]].to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_timeseries(panel)
    plot_scatter(panel)
    plot_by_president(panel)
    print("Done.")


if __name__ == "__main__":
    run()
