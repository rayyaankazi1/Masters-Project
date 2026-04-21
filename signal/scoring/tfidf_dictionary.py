"""
signal/scoring/tfidf_dictionary.py
────────────────────────────────────
Stage 4 of the signal pipeline (README §Pipeline 1 — Stage 4).

Applies the hand-curated hawkish/dovish dictionary to the paragraph-level
corpus and aggregates to speech-level scores.  The LDA fiscal-topic
probability (produced by lda.py) is used as a paragraph weight, so any
changes to the LDA pipeline — number of topics, fiscal topic id, probability
threshold — flow through automatically when you:

    1. Re-run signal/topic_modeling/lda.py   → regenerates paragraphs_lda.csv
    2. Re-run this script                    → picks up updated fiscal weights

This script does NOT import or re-run any LDA code.  It reads only
data/interim/paragraphs_lda.csv and the dictionary files.

Scoring formula
───────────────
For each paragraph p:
    hawkish_tf_p = Σ count(term_i in p) / n_tokens_p   (hawkish terms)
    dovish_tf_p  = Σ count(term_i in p) / n_tokens_p   (dovish terms)
    net_tf_p     = hawkish_tf_p − dovish_tf_p

Aggregated to speech level using two weighting schemes:
    Primary   — weight_p = fiscal_topic_prob_p × n_tokens_p
                (fiscal-relevant AND longer paragraphs count more)
    Robustness — equal weight across all paragraphs in speech

Reads
─────
    data/interim/paragraphs_lda.csv
    signal/dictionaries/hawkish_terms.txt
    signal/dictionaries/dovish_terms.txt

Writes
──────
    data/interim/paragraphs_scored.csv    paragraph-level with hit counts
    data/interim/speeches_scored.csv      speech-level aggregated scores
    outputs/figures/scoring_overview.png  summary charts
    outputs/tables/scoring_summary.txt    printable summary
"""

import os
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_CSV     = os.path.join(_ROOT, "data", "interim", "paragraphs_lda.csv")
DICT_DIR     = os.path.join(_ROOT, "signal", "dictionaries")
HAWKISH_TXT  = os.path.join(DICT_DIR, "hawkish_terms.txt")
DOVISH_TXT   = os.path.join(DICT_DIR, "dovish_terms.txt")
INTERIM_DIR  = os.path.join(_ROOT, "data", "interim")
FIGURES_DIR  = os.path.join(_ROOT, "outputs", "figures")
TABLES_DIR   = os.path.join(_ROOT, "outputs", "tables")

# ── Config ────────────────────────────────────────────────────────────────────
PRES_ORDER  = ["Macri", "AF", "Milei"]
PRES_COLORS = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}

# ── Text normalisation ────────────────────────────────────────────────────────
_CLEAN_RE = re.compile(r"[^a-z\s]")

def normalise(text: str) -> str:
    """Lowercase, strip accents, remove non-alpha characters."""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return _CLEAN_RE.sub(" ", text)

# ── Dictionary loader ─────────────────────────────────────────────────────────

def load_dictionary(path: str) -> list[str]:
    """
    Load terms from a .txt file.
    Lines starting with # are comments and are ignored.
    Returns a list of normalised terms sorted longest-first
    (so multi-word phrases are matched before their component words).
    """
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(normalise(line))
    # Sort longest first so "deficit cero" is matched before "deficit"
    return sorted(set(terms), key=len, reverse=True)

# ── Paragraph-level scoring ───────────────────────────────────────────────────

def score_paragraph(
    text: str,
    n_tokens: int,
    hawkish_terms: list[str],
    dovish_terms: list[str],
) -> dict:
    """
    Count hawkish and dovish term hits in a single normalised paragraph.
    Returns raw counts and normalised term frequencies.
    """
    norm = normalise(text)

    hawkish_hits = {t: norm.count(t) for t in hawkish_terms if norm.count(t) > 0}
    dovish_hits  = {t: norm.count(t) for t in dovish_terms  if norm.count(t) > 0}

    h_count = sum(hawkish_hits.values())
    d_count = sum(dovish_hits.values())
    denom   = max(n_tokens, 1)

    h_tf = h_count / denom
    d_tf = d_count / denom
    net  = h_tf - d_tf
    total = h_tf + d_tf

    return {
        "hawkish_hits":  h_count,
        "dovish_hits":   d_count,
        "hawkish_terms_matched": list(hawkish_hits.keys()),
        "dovish_terms_matched":  list(dovish_hits.keys()),
        "hawkish_tf":    h_tf,
        "dovish_tf":     d_tf,
        "net_tf":        net,
        "tone_index":    (net / total) if total > 0 else np.nan,
    }

# ── Speech-level aggregation ──────────────────────────────────────────────────

def aggregate_to_speech(para_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate paragraph-level scores to speech level using two schemes:

    Primary   — fiscal_topic_prob × n_tokens weighted average
    Robustness — equal-weight average across all paragraphs

    The primary scheme gives more weight to paragraphs that are both
    fiscal-relevant (high fiscal_topic_prob) and substantive (long).
    """
    records = []

    for speech_id, grp in para_df.groupby("speech_id"):
        meta = grp.iloc[0]

        # Weights: fiscal probability × paragraph length
        weights_fiscal = grp["fiscal_topic_prob"] * grp["n_tokens"]
        w_sum = weights_fiscal.sum()

        if w_sum > 0:
            h_tf_w = (grp["hawkish_tf"] * weights_fiscal).sum() / w_sum
            d_tf_w = (grp["dovish_tf"]  * weights_fiscal).sum() / w_sum
        else:
            h_tf_w = d_tf_w = np.nan

        net_tf_w = h_tf_w - d_tf_w if not np.isnan(h_tf_w) else np.nan
        total_w  = h_tf_w + d_tf_w if not np.isnan(h_tf_w) else np.nan

        # Robustness: equal weight
        h_tf_eq = grp["hawkish_tf"].mean()
        d_tf_eq = grp["dovish_tf"].mean()
        net_tf_eq = h_tf_eq - d_tf_eq

        records.append({
            "speech_id":           speech_id,
            "date":                meta["date"],
            "president":           meta["president"],
            "president_id":        meta["president_id"],
            "year_month":          meta["year_month"],
            "n_paragraphs":        len(grp),
            "n_fiscal_paragraphs": grp["is_fiscal"].sum(),
            "fiscal_weight_sum":   w_sum,
            "hawkish_hits_total":  grp["hawkish_hits"].sum(),
            "dovish_hits_total":   grp["dovish_hits"].sum(),
            # Primary score (fiscal-probability × word-count weighted)
            "hawkish_tf_weighted": h_tf_w,
            "dovish_tf_weighted":  d_tf_w,
            "net_tf_weighted":     net_tf_w,
            "tone_index_weighted": (net_tf_w / total_w)
                                   if (total_w is not None and total_w > 0) else np.nan,
            # Robustness score (equal weight)
            "hawkish_tf_equal":    h_tf_eq,
            "dovish_tf_equal":     d_tf_eq,
            "net_tf_equal":        net_tf_eq,
        })

    speech_df = pd.DataFrame(records)
    speech_df["date"] = pd.to_datetime(speech_df["date"])
    speech_df.sort_values("date", inplace=True, ignore_index=True)
    return speech_df

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scoring_overview(speech_df: pd.DataFrame):
    core = speech_df[speech_df["president"].isin(PRES_ORDER)].copy()
    core["ym_dt"] = pd.to_datetime(core["year_month"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # ── Panel 1: monthly mean net_tf_weighted over time ───────────────────────
    monthly = (
        core.groupby(["year_month", "president"], observed=True)["net_tf_weighted"]
        .mean()
        .reset_index()
    )
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])

    ax = axes[0]
    for pres in PRES_ORDER:
        sub = monthly[monthly["president"] == pres].sort_values("ym_dt")
        ax.plot(sub["ym_dt"], sub["net_tf_weighted"],
                label=pres, color=PRES_COLORS[pres], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for date in ["2015-12-10", "2019-12-10", "2023-12-10"]:
        ax.axvline(pd.Timestamp(date), color="grey", linewidth=1,
                   linestyle="--", alpha=0.5)
    ax.set_title("Monthly mean hawkishness score (net TF, fiscal-probability weighted)")
    ax.set_ylabel("Net TF score")
    ax.legend()

    # ── Panel 2: distribution of speech-level scores by president ─────────────
    ax = axes[1]
    data = [core[core["president"] == p]["net_tf_weighted"].dropna().values
            for p in PRES_ORDER]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, pres in zip(bp["boxes"], PRES_ORDER):
        patch.set_facecolor(PRES_COLORS[pres])
    ax.set_xticklabels(PRES_ORDER)
    ax.set_ylabel("Net TF score (per speech)")
    ax.set_title("Distribution of speech-level hawkishness scores")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # ── Panel 3: weighted vs equal-weight robustness check ────────────────────
    ax = axes[2]
    ax.scatter(
        core["net_tf_equal"],
        core["net_tf_weighted"],
        c=[PRES_COLORS.get(p, "grey") for p in core["president"]],
        alpha=0.4, s=15,
    )
    lims = [
        min(core["net_tf_equal"].min(), core["net_tf_weighted"].min()) - 0.001,
        max(core["net_tf_equal"].max(), core["net_tf_weighted"].max()) + 0.001,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="45° line")
    ax.set_xlabel("Equal-weight score (robustness)")
    ax.set_ylabel("Fiscal-prob weighted score (primary)")
    ax.set_title("Primary vs robustness score — should be strongly correlated")
    ax.legend(fontsize=8)

    # Legend for president colours
    from matplotlib.patches import Patch
    handles = [Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER]
    axes[2].legend(handles=handles + [plt.Line2D([0],[0],color='k',
                   linestyle='--', label='45° line')], fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "scoring_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    for d in [INTERIM_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load dictionaries ──────────────────────────────────────────────────
    print("Loading dictionaries...")
    hawkish_terms = load_dictionary(HAWKISH_TXT)
    dovish_terms  = load_dictionary(DOVISH_TXT)
    print(f"  Hawkish terms : {len(hawkish_terms)}")
    print(f"  Dovish terms  : {len(dovish_terms)}")

    # ── 2. Load paragraph dataframe from LDA output ───────────────────────────
    print(f"\nLoading {PARA_CSV}...")
    if not os.path.exists(PARA_CSV):
        raise FileNotFoundError(
            "paragraphs_lda.csv not found. Run signal/topic_modeling/lda.py first."
        )
    para_df = pd.read_csv(PARA_CSV)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)].copy()
    print(f"  {len(para_df):,} paragraphs loaded")
    print(f"  LDA fiscal column: fiscal_topic_prob  "
          f"(min={para_df['fiscal_topic_prob'].min():.3f}, "
          f"max={para_df['fiscal_topic_prob'].max():.3f})")

    # ── 3. Score each paragraph ───────────────────────────────────────────────
    print("\nScoring paragraphs...")
    score_records = []
    for _, row in para_df.iterrows():
        scores = score_paragraph(
            str(row["text_para"]),
            int(row["n_tokens"]),
            hawkish_terms,
            dovish_terms,
        )
        score_records.append(scores)

    score_df = pd.DataFrame(score_records)
    para_scored = pd.concat(
        [para_df.reset_index(drop=True), score_df.reset_index(drop=True)],
        axis=1,
    )

    any_hits = (para_scored["hawkish_hits"] + para_scored["dovish_hits"]) > 0
    print(f"  Paragraphs with ≥1 dictionary hit: "
          f"{any_hits.sum():,} / {len(para_scored):,} "
          f"({any_hits.mean()*100:.1f}%)")
    print(f"  Total hawkish hits: {para_scored['hawkish_hits'].sum():,}")
    print(f"  Total dovish hits:  {para_scored['dovish_hits'].sum():,}")

    # ── 4. Aggregate to speech level ──────────────────────────────────────────
    print("\nAggregating to speech level...")
    speech_df = aggregate_to_speech(para_scored)
    print(f"  {len(speech_df):,} speeches scored")

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    para_out = os.path.join(INTERIM_DIR, "paragraphs_scored.csv")
    para_scored.drop(columns=["hawkish_terms_matched", "dovish_terms_matched"],
                     errors="ignore").to_csv(para_out, index=False)
    print(f"\nSaved: {para_out}")

    speech_out = os.path.join(INTERIM_DIR, "speeches_scored.csv")
    speech_df.to_csv(speech_out, index=False)
    print(f"Saved: {speech_out}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_scoring_overview(speech_df)

    summary_lines = [
        "=== DICTIONARY SCORING SUMMARY ===",
        f"Hawkish terms : {len(hawkish_terms)}",
        f"Dovish terms  : {len(dovish_terms)}",
        "",
        "Paragraphs with hits:",
        f"  Any hit   : {any_hits.sum():,} / {len(para_scored):,} "
        f"({any_hits.mean()*100:.1f}%)",
        f"  Hawkish   : {(para_scored['hawkish_hits']>0).sum():,}",
        f"  Dovish    : {(para_scored['dovish_hits']>0).sum():,}",
        "",
        "Speech-level net_tf_weighted by president:",
        (
            speech_df[speech_df["president"].isin(PRES_ORDER)]
            .groupby("president")["net_tf_weighted"]
            .describe()[["mean", "std", "min", "50%", "max"]]
            .loc[PRES_ORDER]
            .round(6)
            .to_string()
        ),
        "",
        "Top 15 hawkish terms by total hits:",
    ]

    # Top terms
    hit_counts = {}
    for _, row in para_scored.iterrows():
        norm = normalise(str(row["text_para"]))
        for t in hawkish_terms:
            c = norm.count(t)
            if c > 0:
                hit_counts[t] = hit_counts.get(t, 0) + c
    top_hawkish = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for term, count in top_hawkish:
        summary_lines.append(f"  {count:5d}  {term}")

    summary_lines += ["", "Top 15 dovish terms by total hits:"]
    hit_counts_d = {}
    for _, row in para_scored.iterrows():
        norm = normalise(str(row["text_para"]))
        for t in dovish_terms:
            c = norm.count(t)
            if c > 0:
                hit_counts_d[t] = hit_counts_d.get(t, 0) + c
    top_dovish = sorted(hit_counts_d.items(), key=lambda x: x[1], reverse=True)[:15]
    for term, count in top_dovish:
        summary_lines.append(f"  {count:5d}  {term}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(TABLES_DIR, "scoring_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {summary_path}")

    return para_scored, speech_df


if __name__ == "__main__":
    para_scored, speech_df = run()
