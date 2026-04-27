"""
signal/scoring/tfidf_dictionary.py
────────────────────────────────────
Stage 4 of the signal pipeline (README §Pipeline 1 — Stage 4).

Applies the hand-curated hawkish/dovish dictionary to the paragraph-level
corpus and aggregates to speech-level scores.  The LDA fiscal-topic
probability (produced by lda.py) is used as a paragraph weight, so any
changes to the LDA pipeline flow through automatically when you:

    1. Re-run signal/topic_modeling/lda.py   → regenerates paragraphs_lda.csv
    2. Re-run this script                    → picks up updated fiscal weights

This script does NOT import or re-run any LDA code.

Improvements (v5)
─────────────────
1. Regex morphological matching — each word in a term gets a \\w* suffix.
2. Sentence-level scoring — text split into sentences before matching.
3. Zero-hit dead terms pruned; verb/stem forms added.
4. Dictionary expanded with literature-grounded terms.
5. Dual LDA weighting: max(fiscal_topic_prob, monetary_topic_prob).
6. Negation-aware scoring — before counting a match, a word window before
   the match position is inspected for critical/ironic usage signals:

   Tier 1 — strong signals (10-word window):
     supuest*, llamad*, rechaz*, critic*, elimin*, en contra, opuest*,
     en nombre de, lo que llaman, mal llamad*, negar*, combat*
   Tier 2 — weak signals (3-word window):
     no, nunca, jamas

   Negated hits are tracked separately (hawkish_negated, dovish_negated)
   and reported in scoring_summary.txt for validation.  Negated hits are
   NOT counted toward the score or flipped in direction.

   Motivation: Milei uses "justicia social", "gradualismo", "redistribucion"
   etc. constantly to criticise or mock them (libertarian lectures,
   Hoppe quotes).  Without negation detection, ~400 false dovish hits
   inflate his dovish TF score.

7. Direction-aware elimin* (v5 — negation audit 2026-04):
   The elimin* Tier-1 pattern is now gated by match direction.
   It suppresses dovish hits correctly ("eliminar la redistribución") but
   no longer suppresses hawkish hits ("para eliminar la inflación, mantener
   el superávit fiscal"; "la eliminación del impuesto inflacionario").
   Corpus audit: 11 false hawkish suppressions corrected (all Milei).
   is_negated() now takes an optional direction="hawkish"|"dovish" parameter.

   Documented residual false suppressions (~11 hits, <0.7% of gross):
   • "no solo X" (additive) — Tier-2 fires on "no" in "no solo" (4 hits)
   • "no hay plata" self-negation — term starts with "no", 2nd occurrence
     in same sentence sees "no" in 3-word window (3 hits, all Milei)
   • Historical "nunca viste/tuvimos X" — Milei contrast-framing his own
     achievement; "nunca" modifies verb, not the fiscal concept (4 hits)
   These are known limitations, not coded around (signal impact < 1%).

Scoring formula
───────────────
For each paragraph p, split into sentences s_1 … s_k:
    For each sentence s_i find regex matches with finditer.
    For each match call is_negated(sentence, match.start()).
    Only non-negated matches are counted.
    hawkish_tf_p = Σ non-negated hits / n_tokens_p
    dovish_tf_p  = Σ non-negated hits / n_tokens_p
    net_tf_p     = hawkish_tf_p − dovish_tf_p

Aggregated to speech level using two weighting schemes:
    Primary   — weight_p = max(fiscal_prob, monetary_prob) × n_tokens_p
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
    data/interim/monthly_signal.csv       monthly BVAR-ready signal (Method C)
    outputs/figures/scoring_overview.png  summary charts
    outputs/tables/scoring_summary.txt    printable summary
"""

import os
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# LDA dual-weight: set MONETARY_TOPIC_ID to None to use fiscal prob only.
# With k=10, topic 8 top words: dinero, precios, tasa, cambio, mercado.
MONETARY_TOPIC_ID = 8

# ── Negation detection ────────────────────────────────────────────────────────
# Window sizes (in words) to inspect BEFORE a match start position.
NEG_WINDOW_STRONG = 10   # for strong critical-framing signals
NEG_WINDOW_WEAK   = 3    # for bare negation words (no, nunca, jamas)

# Tier 1 — strong signals: ironic distance, opposition verbs, critical framing.
# Applied to the NEG_WINDOW_STRONG words immediately before the match.
_NEG_STRONG = [re.compile(p) for p in [
    r"\bsupuest\w*\b",      # supuesta/o — "la supuesta justicia social"
    r"\bllamad\w*\b",       # llamada/o  — "la llamada redistribucion"
    r"\bmal\s+llamad\w*\b", # mal llamada/o
    r"\brechaz\w*\b",       # rechazo/an/amos — "rechazo el gradualismo"
    r"\bcritic\w*\b",       # critico/a/amos  — "critico la intervencion"
    r"\belimin\w*\b",       # eliminar/o/amos — "eliminar la redistribucion"
    r"\ben\s+contra\b",     # "en contra de"
    r"\bopuest\w*\b",       # opuesto/a       — "me opongo / opuesto a"
    r"\ben\s+nombre\s+de\b",# "en nombre de la justicia social"
    r"\blo\s+que\s+llaman\b",# "lo que llaman justicia social"
    r"\bso\s+pretexto\b",   # "so pretexto de"
    r"\bnegar\w*\b",        # negar/nego/negamos
    r"\bcombat\w*\b",       # combatir/o/imos — "combatir la intervencion"
    r"\bfals\w*\b",         # falsa/o         — "la falsa justicia"
    r"\bideolog\w*\b",      # ideologia/ico  — Milei's framing of these as ideology
]]

# Tier 2 — weak signals: bare negation words (shorter window to avoid
# false negatives like "no podemos ignorar la justicia social").
_NEG_WEAK = [re.compile(p) for p in [
    r"\bno\b",
    r"\bnunca\b",
    r"\bjamas\b",
]]


def is_negated(sentence: str, match_start: int, direction: str = "both") -> bool:
    """
    Return True if the match at `match_start` in `sentence` is preceded by
    negation or critical-usage signals within the configured word windows.

    Parameters
    ----------
    sentence    : normalised sentence text
    match_start : character start position of the regex match
    direction   : "hawkish" | "dovish" | "both"
                  Controls which Tier-1 patterns are applied.
                  The `elimin*` pattern is direction-gated: it is designed to
                  suppress dovish terms used in a "we must eliminate X" context
                  (e.g. Milei — "eliminar la redistribución"), but should NOT
                  suppress hawkish terms in the same construction (e.g. Milei —
                  "para eliminar la inflación, mantener el superávit fiscal",
                  "la eliminación del impuesto inflacionario").
                  When direction == "hawkish", elimin* is skipped.

    Uses two tiers:
      - Strong patterns checked over the last NEG_WINDOW_STRONG words.
      - Weak patterns checked over the last NEG_WINDOW_WEAK words only.

    Operates on normalised text (no accents, lowercase, no punctuation).
    """
    pre_words = sentence[:match_start].split()

    # Tier 1: strong critical-framing signals
    window_strong = " ".join(pre_words[-NEG_WINDOW_STRONG:])
    for pat in _NEG_STRONG:
        # Skip elimin* for hawkish terms — see docstring.
        if pat.pattern == r"\belimin\w*\b" and direction == "hawkish":
            continue
        if pat.search(window_strong):
            return True

    # Tier 2: bare negation — tight window to reduce false positives
    window_weak = " ".join(pre_words[-NEG_WINDOW_WEAK:])
    if any(p.search(window_weak) for p in _NEG_WEAK):
        return True

    return False


# ── Text normalisation ────────────────────────────────────────────────────────
_CLEAN_RE  = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")

def normalise(text: str) -> str:
    """Lowercase, strip accents, remove non-alpha, collapse whitespace."""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = _CLEAN_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


# ── Sentence splitting ────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?,;:])\s+")

def split_sentences(text: str) -> list[str]:
    """Split normalised paragraph text into approximate sentences."""
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts if parts else [text]


# ── Regex pattern builder ─────────────────────────────────────────────────────

def term_to_pattern(term: str) -> re.Pattern:
    """
    Compile a normalised term into a regex pattern with morphological
    flexibility.  Each word gets a \\w* suffix.

    "ajuste fiscal"  →  r"ajuste\\w*\\s+fiscal\\w*"
    "privatizar"     →  r"privatizar\\w*"
    """
    words = term.split()
    pattern_str = r"\s+".join(w + r"\w*" for w in words)
    return re.compile(pattern_str)


# ── Dictionary loader ─────────────────────────────────────────────────────────

def load_dictionary(path: str) -> list[tuple[str, re.Pattern]]:
    """
    Load terms from a .txt file and compile each to a regex pattern.
    Returns list of (term, pattern) sorted longest-first.
    """
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(normalise(line))
    unique_terms = sorted(set(terms), key=len, reverse=True)
    return [(t, term_to_pattern(t)) for t in unique_terms]


# ── Paragraph-level scoring ───────────────────────────────────────────────────

def score_paragraph(
    text: str,
    n_tokens: int,
    hawkish_patterns: list[tuple[str, re.Pattern]],
    dovish_patterns: list[tuple[str, re.Pattern]],
) -> dict:
    """
    Count hawkish and dovish regex hits at sentence level with negation
    detection.  Each match is tested by is_negated() before being counted.
    Negated hits are tallied separately and excluded from TF scores.

    Returns
    -------
    dict with hit counts, negated counts, TF scores, matched term lists,
    negated term lists, and tone index.
    """
    norm = normalise(text)
    sentences = split_sentences(norm)

    h_term_counts: dict[str, int] = {}
    d_term_counts: dict[str, int] = {}
    h_neg_counts:  dict[str, int] = {}
    d_neg_counts:  dict[str, int] = {}

    for sent in sentences:
        for term, pat in hawkish_patterns:
            for m in pat.finditer(sent):
                if is_negated(sent, m.start(), direction="hawkish"):
                    h_neg_counts[term] = h_neg_counts.get(term, 0) + 1
                else:
                    h_term_counts[term] = h_term_counts.get(term, 0) + 1

        for term, pat in dovish_patterns:
            for m in pat.finditer(sent):
                if is_negated(sent, m.start(), direction="dovish"):
                    d_neg_counts[term] = d_neg_counts.get(term, 0) + 1
                else:
                    d_term_counts[term] = d_term_counts.get(term, 0) + 1

    h_count     = sum(h_term_counts.values())
    d_count     = sum(d_term_counts.values())
    h_neg_count = sum(h_neg_counts.values())
    d_neg_count = sum(d_neg_counts.values())
    denom       = max(n_tokens, 1)

    h_tf  = h_count / denom
    d_tf  = d_count / denom
    net   = h_tf - d_tf
    total = h_tf + d_tf

    return {
        "hawkish_hits":           h_count,
        "dovish_hits":            d_count,
        "hawkish_negated":        h_neg_count,
        "dovish_negated":         d_neg_count,
        "hawkish_terms_matched":  list(h_term_counts.keys()),
        "dovish_terms_matched":   list(d_term_counts.keys()),
        "hawkish_terms_negated":  list(h_neg_counts.keys()),
        "dovish_terms_negated":   list(d_neg_counts.keys()),
        "hawkish_tf":             h_tf,
        "dovish_tf":              d_tf,
        "net_tf":                 net,
        "tone_index":             (net / total) if total > 0 else np.nan,
    }


# ── Speech-level aggregation ──────────────────────────────────────────────────

def aggregate_to_speech(para_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate paragraph-level scores to speech level.

    Primary   — max(fiscal_topic_prob, monetary_topic_prob) × n_tokens
    Robustness — equal-weight average across all paragraphs
    """
    if MONETARY_TOPIC_ID is not None:
        mon_col = f"topic_{MONETARY_TOPIC_ID}"
        if mon_col in para_df.columns:
            combined_prob = para_df[["fiscal_topic_prob", mon_col]].max(axis=1)
        else:
            combined_prob = para_df["fiscal_topic_prob"]
    else:
        combined_prob = para_df["fiscal_topic_prob"]

    records = []

    for speech_id, grp in para_df.groupby("speech_id"):
        meta     = grp.iloc[0]
        grp_prob = combined_prob.loc[grp.index]

        weights  = grp_prob * grp["n_tokens"]
        w_sum    = weights.sum()

        if w_sum > 0:
            h_tf_w = (grp["hawkish_tf"] * weights).sum() / w_sum
            d_tf_w = (grp["dovish_tf"]  * weights).sum() / w_sum
        else:
            h_tf_w = d_tf_w = np.nan

        net_tf_w = h_tf_w - d_tf_w if not np.isnan(h_tf_w) else np.nan
        total_w  = h_tf_w + d_tf_w if not np.isnan(h_tf_w) else np.nan

        h_tf_eq   = grp["hawkish_tf"].mean()
        d_tf_eq   = grp["dovish_tf"].mean()
        net_tf_eq = h_tf_eq - d_tf_eq

        records.append({
            "speech_id":              speech_id,
            "date":                   meta["date"],
            "president":              meta["president"],
            "president_id":           meta["president_id"],
            "year_month":             meta["year_month"],
            "n_paragraphs":           len(grp),
            "n_fiscal_paragraphs":    grp["is_fiscal"].sum(),
            "fiscal_weight_sum":      w_sum,
            "hawkish_hits_total":     grp["hawkish_hits"].sum(),
            "dovish_hits_total":      grp["dovish_hits"].sum(),
            "hawkish_negated_total":  grp["hawkish_negated"].sum(),
            "dovish_negated_total":   grp["dovish_negated"].sum(),
            # Primary score
            "hawkish_tf_weighted":    h_tf_w,
            "dovish_tf_weighted":     d_tf_w,
            "net_tf_weighted":        net_tf_w,
            "tone_index_weighted":    (net_tf_w / total_w)
                                      if (total_w is not None and total_w > 0) else np.nan,
            # Robustness score
            "hawkish_tf_equal":       h_tf_eq,
            "dovish_tf_equal":        d_tf_eq,
            "net_tf_equal":           net_tf_eq,
        })

    speech_df = pd.DataFrame(records)
    speech_df["date"] = pd.to_datetime(speech_df["date"])
    speech_df.sort_values("date", inplace=True, ignore_index=True)

    for col in ["net_tf_weighted", "tone_index_weighted", "net_tf_equal"]:
        mu  = speech_df[col].mean()
        sig = speech_df[col].std()
        speech_df[f"{col}_z"] = (speech_df[col] - mu) / sig if sig > 0 else 0.0

    return speech_df


# ── Monthly aggregation (BVAR input) ─────────────────────────────────────────

def aggregate_to_monthly(speech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate speech-level scores to a monthly series suitable for BVAR.

    Method C — fiscal-weight-sum weighted mean:
        monthly_score = Σ(net_tf_weighted × fiscal_weight_sum) / Σ(fiscal_weight_sum)

    This is a soft fiscal-content filter: non-fiscal speeches (tiny
    fiscal_weight_sum) contribute near-zero weight without being hard-excluded.
    Avoids the small-sample noise of a hard n_fiscal_paragraphs >= 1 filter
    while still recovering ~60% more variance than the naive equal-weight mean.

    Method A (equal-weight mean) is retained as a robustness check column.

    Both series are z-score normalised over the full cross-president sample
    so that Milei's consistently higher level is preserved in the z-scores.
    Use net_tf_fwsum_z as the primary BVAR input.

    Parameters
    ----------
    speech_df : output of aggregate_to_speech(), all presidents.

    Returns
    -------
    pd.DataFrame with one row per (year_month, president), columns:
        year_month, president, n_speeches, n_fiscal_speeches,
        fiscal_weight_total,
        net_tf_fwsum      — Method C raw score
        net_tf_equal_month — Method A raw score (robustness)
        hawkish_tf_fwsum, dovish_tf_fwsum  — components
        net_tf_fwsum_z    — Method C z-score  ← BVAR input
        net_tf_equal_z    — Method A z-score  (robustness)
    """
    core = speech_df[speech_df["president"].isin(PRES_ORDER)].copy()

    records = []
    for (ym, pres), grp in core.groupby(["year_month", "president"], observed=True):
        w     = grp["fiscal_weight_sum"]
        w_sum = w.sum()

        if w_sum > 0:
            net_c  = (grp["net_tf_weighted"]     * w).sum() / w_sum
            h_c    = (grp["hawkish_tf_weighted"]  * w).sum() / w_sum
            d_c    = (grp["dovish_tf_weighted"]   * w).sum() / w_sum
        else:
            net_c = h_c = d_c = np.nan

        records.append({
            "year_month":          ym,
            "president":           pres,
            "n_speeches":          len(grp),
            "n_fiscal_speeches":   (grp["n_fiscal_paragraphs"] >= 1).sum(),
            "fiscal_weight_total": w_sum,
            "net_tf_fwsum":        net_c,
            "hawkish_tf_fwsum":    h_c,
            "dovish_tf_fwsum":     d_c,
            "net_tf_equal_month":  grp["net_tf_weighted"].mean(),
        })

    monthly = pd.DataFrame(records)
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])
    monthly.sort_values(["ym_dt", "president"], inplace=True, ignore_index=True)

    # ── Z-score over full cross-president sample ──────────────────────────────
    for raw_col, z_col in [
        ("net_tf_fwsum",       "net_tf_fwsum_z"),
        ("net_tf_equal_month", "net_tf_equal_z"),
    ]:
        mu  = monthly[raw_col].mean()
        sig = monthly[raw_col].std()
        monthly[z_col] = (monthly[raw_col] - mu) / sig if sig > 0 else 0.0

    monthly.drop(columns=["ym_dt"], inplace=True)
    return monthly


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scoring_overview(speech_df: pd.DataFrame):
    core = speech_df[speech_df["president"].isin(PRES_ORDER)].copy()
    core["ym_dt"] = pd.to_datetime(core["year_month"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    monthly = (
        core.groupby(["year_month", "president"], observed=True)["net_tf_weighted"]
        .mean().reset_index()
    )
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])

    # ── Method C (FW-sum weighted) ── primary BVAR input ─────────────────────
    monthly_c = aggregate_to_monthly(speech_df)
    monthly_c["ym_dt"] = pd.to_datetime(monthly_c["year_month"])

    ax = axes[0]
    for pres in PRES_ORDER:
        sub_a = monthly[monthly["president"] == pres].sort_values("ym_dt")
        sub_c = monthly_c[monthly_c["president"] == pres].sort_values("ym_dt")
        ax.plot(sub_a["ym_dt"], sub_a["net_tf_weighted"],
                color=PRES_COLORS[pres], linewidth=1.0, linestyle="--",
                alpha=0.4, label=f"{pres} (equal-wt)")
        ax.plot(sub_c["ym_dt"], sub_c["net_tf_fwsum"],
                color=PRES_COLORS[pres], linewidth=1.8, linestyle="-",
                label=f"{pres} (FW-sum, BVAR)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for date in ["2015-12-10", "2019-12-10", "2023-12-10"]:
        ax.axvline(pd.Timestamp(date), color="grey", linewidth=1,
                   linestyle="--", alpha=0.5)
    ax.set_title("Monthly hawkishness: Method A equal-weight (dashed) vs Method C FW-sum weighted (solid, BVAR input)")
    ax.set_ylabel("Net TF score")
    ax.legend(fontsize=7, ncol=2)

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

    ax = axes[2]
    ax.scatter(
        core["net_tf_equal"], core["net_tf_weighted"],
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
    ax.set_title("Primary vs robustness score")

    from matplotlib.patches import Patch
    handles = [Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER]
    axes[2].legend(handles=handles + [plt.Line2D([0],[0], color='k',
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
    hawkish_patterns = load_dictionary(HAWKISH_TXT)
    dovish_patterns  = load_dictionary(DOVISH_TXT)
    print(f"  Hawkish terms : {len(hawkish_patterns)}")
    print(f"  Dovish terms  : {len(dovish_patterns)}")

    # ── 2. Load paragraph dataframe ───────────────────────────────────────────
    print(f"\nLoading {PARA_CSV}...")
    if not os.path.exists(PARA_CSV):
        raise FileNotFoundError(
            "paragraphs_lda.csv not found. Run signal/topic_modeling/lda.py first."
        )
    para_df = pd.read_csv(PARA_CSV)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)].copy()
    print(f"  {len(para_df):,} paragraphs loaded")

    # ── 3. Score each paragraph ───────────────────────────────────────────────
    print("\nScoring paragraphs (negation-aware, sentence-level)...")
    score_records = []
    for _, row in para_df.iterrows():
        scores = score_paragraph(
            str(row["text_para"]),
            int(row["n_tokens"]),
            hawkish_patterns,
            dovish_patterns,
        )
        score_records.append(scores)

    score_df    = pd.DataFrame(score_records)
    para_scored = pd.concat(
        [para_df.reset_index(drop=True), score_df.reset_index(drop=True)],
        axis=1,
    )

    any_hits = (para_scored["hawkish_hits"] + para_scored["dovish_hits"]) > 0
    print(f"  Paragraphs with ≥1 accepted hit : "
          f"{any_hits.sum():,} / {len(para_scored):,} "
          f"({any_hits.mean()*100:.1f}%)")
    print(f"  Total hawkish hits (accepted)   : {para_scored['hawkish_hits'].sum():,}")
    print(f"  Total dovish hits  (accepted)   : {para_scored['dovish_hits'].sum():,}")
    print(f"  Total hawkish negated (rejected): {para_scored['hawkish_negated'].sum():,}")
    print(f"  Total dovish negated  (rejected): {para_scored['dovish_negated'].sum():,}")

    # ── 4. Aggregate to speech level ──────────────────────────────────────────
    print("\nAggregating to speech level...")
    speech_df = aggregate_to_speech(para_scored)
    print(f"  {len(speech_df):,} speeches scored")

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    drop_cols = [
        "hawkish_terms_matched", "dovish_terms_matched",
        "hawkish_terms_negated", "dovish_terms_negated",
    ]
    para_out = os.path.join(INTERIM_DIR, "paragraphs_scored.csv")
    para_scored.drop(columns=drop_cols, errors="ignore").to_csv(para_out, index=False)
    print(f"\nSaved: {para_out}")

    speech_out = os.path.join(INTERIM_DIR, "speeches_scored.csv")
    speech_df.to_csv(speech_out, index=False)
    print(f"Saved: {speech_out}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_scoring_overview(speech_df)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    # Re-scan for per-term, per-president counts (accepted + negated)
    hit_h:     dict[str, int] = {}
    hit_d:     dict[str, int] = {}
    neg_h:     dict[str, int] = {}
    neg_d:     dict[str, int] = {}
    # Per-president negation breakdown
    pres_neg: dict[str, dict] = {
        p: {"h_acc": 0, "d_acc": 0, "h_neg": 0, "d_neg": 0}
        for p in PRES_ORDER
    }

    for _, row in para_scored.iterrows():
        pres = row["president"]
        norm = normalise(str(row["text_para"]))
        sents = split_sentences(norm)

        for term, pat in hawkish_patterns:
            for sent in sents:
                for m in pat.finditer(sent):
                    if is_negated(sent, m.start(), direction="hawkish"):
                        neg_h[term] = neg_h.get(term, 0) + 1
                        if pres in pres_neg:
                            pres_neg[pres]["h_neg"] += 1
                    else:
                        hit_h[term] = hit_h.get(term, 0) + 1
                        if pres in pres_neg:
                            pres_neg[pres]["h_acc"] += 1

        for term, pat in dovish_patterns:
            for sent in sents:
                for m in pat.finditer(sent):
                    if is_negated(sent, m.start(), direction="dovish"):
                        neg_d[term] = neg_d.get(term, 0) + 1
                        if pres in pres_neg:
                            pres_neg[pres]["d_neg"] += 1
                    else:
                        hit_d[term] = hit_d.get(term, 0) + 1
                        if pres in pres_neg:
                            pres_neg[pres]["d_acc"] += 1

    summary_lines = [
        "=== DICTIONARY SCORING SUMMARY (v4 — negation-aware) ===",
        f"Hawkish terms : {len(hawkish_patterns)}",
        f"Dovish terms  : {len(dovish_patterns)}",
        "",
        "Paragraphs with hits:",
        f"  Any accepted hit : {any_hits.sum():,} / {len(para_scored):,} "
        f"({any_hits.mean()*100:.1f}%)",
        f"  Hawkish accepted : {(para_scored['hawkish_hits']>0).sum():,}",
        f"  Dovish  accepted : {(para_scored['dovish_hits']>0).sum():,}",
        f"  Hawkish negated  : {para_scored['hawkish_negated'].sum():,}",
        f"  Dovish  negated  : {para_scored['dovish_negated'].sum():,}",
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
        "── NEGATION SUPPRESSION REPORT ──────────────────────────────────",
    ]

    for pres in PRES_ORDER:
        d = pres_neg[pres]
        h_gross = d["h_acc"] + d["h_neg"]
        dov_gross = d["d_acc"] + d["d_neg"]
        h_pct = d["h_neg"] / h_gross * 100 if h_gross > 0 else 0
        dov_pct = d["d_neg"] / dov_gross * 100 if dov_gross > 0 else 0
        summary_lines.append(
            f"  {pres:6s}  hawkish: {d['h_acc']:4d} accepted, {d['h_neg']:4d} negated "
            f"({h_pct:.1f}% suppressed)  |  "
            f"dovish: {d['d_acc']:4d} accepted, {d['d_neg']:4d} negated "
            f"({dov_pct:.1f}% suppressed)"
        )

    summary_lines += [
        "",
        "Top negated dovish terms (all presidents):",
    ]
    for term, cnt in sorted(neg_d.items(), key=lambda x: x[1], reverse=True)[:15]:
        summary_lines.append(f"  {cnt:5d}  {term}")

    summary_lines += [
        "",
        "Top negated hawkish terms (all presidents):",
    ]
    for term, cnt in sorted(neg_h.items(), key=lambda x: x[1], reverse=True)[:10]:
        summary_lines.append(f"  {cnt:5d}  {term}")

    summary_lines += [
        "",
        "── TOP ACCEPTED TERMS ────────────────────────────────────────────",
        "Top 15 hawkish terms (accepted hits):",
    ]
    for term, cnt in sorted(hit_h.items(), key=lambda x: x[1], reverse=True)[:15]:
        summary_lines.append(f"  {cnt:5d}  {term}")

    summary_lines += ["", "Top 15 dovish terms (accepted hits):"]
    for term, cnt in sorted(hit_d.items(), key=lambda x: x[1], reverse=True)[:15]:
        summary_lines.append(f"  {cnt:5d}  {term}")

    # Zero-hit diagnostics
    zero_h = sorted({t for t, _ in hawkish_patterns} - set(hit_h.keys()))
    zero_d = sorted({t for t, _ in dovish_patterns}  - set(hit_d.keys()))
    summary_lines += [
        "",
        f"ZERO-HIT HAWKISH ({len(zero_h)}):",
    ]
    for t in zero_h:
        summary_lines.append(f"  {t}")
    summary_lines += [
        "",
        f"ZERO-HIT DOVISH ({len(zero_d)}):",
    ]
    for t in zero_d:
        summary_lines.append(f"  {t}")

    # ── 8. Monthly aggregation (Method C — BVAR input) ───────────────────────
    print("\nAggregating to monthly (Method C — FW-sum weighted)...")
    monthly_df = aggregate_to_monthly(speech_df)

    monthly_out = os.path.join(INTERIM_DIR, "monthly_signal.csv")
    monthly_df.to_csv(monthly_out, index=False)
    print(f"Saved: {monthly_out}")
    print(f"  {len(monthly_df)} month-president rows")

    monthly_stats_lines = [
        "",
        "── MONTHLY SIGNAL STATS (Method C — FW-sum weighted) ────────────────",
        f"{'President':<8} {'N months':>9} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}",
        "-" * 58,
    ]
    for pres in PRES_ORDER:
        sub = monthly_df[monthly_df["president"] == pres]["net_tf_fwsum"].dropna()
        monthly_stats_lines.append(
            f"{pres:<8} {len(sub):>9d} {sub.mean():>10.5f} {sub.std():>10.5f} "
            f"{sub.min():>10.5f} {sub.max():>10.5f}"
        )
    monthly_stats_lines += [
        "",
        "Z-score column (BVAR input): net_tf_fwsum_z",
        f"{'President':<8} {'Mean-z':>10} {'Std-z':>10}",
        "-" * 32,
    ]
    for pres in PRES_ORDER:
        sub = monthly_df[monthly_df["president"] == pres]["net_tf_fwsum_z"].dropna()
        monthly_stats_lines.append(
            f"{pres:<8} {sub.mean():>10.4f} {sub.std():>10.4f}"
        )

    summary_lines += monthly_stats_lines

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(TABLES_DIR, "scoring_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {summary_path}")

    return para_scored, speech_df, monthly_df


if __name__ == "__main__":
    para_scored, speech_df, monthly_df = run()
