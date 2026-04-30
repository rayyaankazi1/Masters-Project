"""
signal/scoring/tfidf_dictionary.py  (v7 — keyword fiscal filter)
────────────────────────────────────────────────────────────────────────
Stage 4 of the signal pipeline.

Signal construction follows Baker, Bloom & Davis (2016) EPU methodology:

    Stage 1 — Keyword fiscal filter (this script)
        A paragraph is labelled fiscal if it contains at least one term from
        FISCAL_KEYWORDS — a short list of core fiscal-policy vocabulary drawn
        directly from the Baker–Bloom–Davis (2016) article-filtering approach.
        This replaces the v6 LDA-threshold filter.

        Rationale: the LDA filter (v6) created an asymmetric bias — it captured
        ~84% of hawkish hits but only ~40% of dovish hits, because hawkish terms
        are inherently fiscal-accounting vocabulary (deficit, superavit) while
        dovish terms include social-spending vocabulary (obra publica, tarjeta
        alimentar) that the LDA assigned to a separate welfare topic.
        A keyword filter is symmetric: it admits paragraphs that discuss fiscal
        policy in either direction.  LDA topic probabilities are retained in
        paragraphs_scored.csv for use as a validation/narrative exhibit.

    Stage 2 — Dictionary direction (this script)
        Within fiscal paragraphs only, each paragraph casts a directional vote:
            has_hawkish = 1  if ≥1 accepted hawkish term present
            has_dovish  = 1  if ≥1 accepted dovish term present
        A paragraph may vote both hawkish and dovish simultaneously.

    Stage 3 — Monthly aggregation
        For each president-month t:
            H_t = Σ has_hawkish   (hawkish-hit fiscal paragraphs)
            D_t = Σ has_dovish    (dovish-hit fiscal paragraphs)
            P_t = total fiscal paragraphs in month t

            signal_t = (H_t − D_t) / P_t

        This is the net hawkish share of fiscal discourse — directly analogous
        to the EPU article-count share.  Normalising by P_t controls for
        presidents who give more speeches in a given month.

    Stage 4 — Z-score normalisation
        Computed over the full cross-president sample so that Milei's higher
        level is preserved in the z-scores (not washed out by within-president
        normalisation).

        Primary BVAR column: net_hawkish_z  in monthly_signal.csv

Negation detection (unchanged from v5/v6)
──────────────────────────────────────────
Before counting a dictionary hit, a word-window before the match is scanned
for critical/ironic usage signals (Tier 1, 10-word window) and bare negation
words (Tier 2, 3-word window).  Negated hits are tracked separately and
reported in scoring_summary.txt for audit.  elimin* is direction-aware:
it suppresses dovish hits but not hawkish ones (see is_negated() docstring).

Reads
──────
    data/interim/paragraphs_lda.csv          (produced by lda.py)
    signal/dictionaries/hawkish_terms.txt
    signal/dictionaries/dovish_terms.txt

Writes
──────
    data/interim/paragraphs_scored.csv       paragraph-level with hit flags
    data/interim/speeches_scored.csv         speech-level counts
    data/interim/monthly_signal.csv          monthly BVAR-ready signal
    data/processed/bvar_signal.csv           clean BVAR-ready output
    outputs/figures/scoring_overview.png     summary charts
    outputs/tables/scoring_summary.txt       audit report
"""

import os
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_CSV    = os.path.join(_ROOT, "data", "interim", "paragraphs_lda.csv")
DICT_DIR    = os.path.join(_ROOT, "signal", "dictionaries")
HAWKISH_TXT = os.path.join(DICT_DIR, "hawkish_terms.txt")
DOVISH_TXT  = os.path.join(DICT_DIR, "dovish_terms.txt")
INTERIM_DIR = os.path.join(_ROOT, "data", "interim")
FIGURES_DIR = os.path.join(_ROOT, "outputs", "figures")
TABLES_DIR  = os.path.join(_ROOT, "outputs", "tables")

# ── Config ────────────────────────────────────────────────────────────────────
PRES_ORDER  = ["Macri", "AF", "Milei"]
PRES_COLORS = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}

# ── Keyword fiscal filter (v7) ────────────────────────────────────────────────
# A paragraph is classified as fiscal if it contains at least one of these
# terms.  Follows Baker, Bloom & Davis (2016): fiscal relevance is determined
# by keyword presence, not a topic model.
#
# Design principles:
#   1. Symmetric — list covers both hawkish fiscal vocabulary (deficit,
#      ajuste, presupuesto) and dovish fiscal vocabulary (obra publica,
#      inversion publica, subsidio) so neither direction is disadvantaged.
#   2. Short and auditable — each term is immediately legible; no model
#      artefacts or threshold choices.
#   3. Directly citable — Baker et al. (2016) Appendix B uses the same
#      keyword-in-text approach for their EPU fiscal policy category.
#
# LDA fiscal_topic_prob is retained in the data for validation purposes.
FISCAL_KEYWORDS = [
    # Core fiscal / budget vocabulary
    "deficit",
    "superavit",
    "gasto",          # gasto* matches gasto, gastos, gasto publico etc.
    "presupuest",     # presupuesto, presupuestario, presupuestaria
    "fiscal",
    "impuesto",       # impuesto, impuestos, impositivo
    "deuda",
    "inflacion",
    "ajuste",
    "subsidio",       # subsidio, subsidios
    "recaudacion",
    "austeridad",
    # Public investment / social spending (ensures dovish content is captured)
    "obra publica",
    "inversion publica",
    "inversion social",
]

# ── Negation detection ────────────────────────────────────────────────────────
NEG_WINDOW_STRONG = 10   # words before match: critical-framing signals
NEG_WINDOW_WEAK   = 3    # words before match: bare negation words

_NEG_STRONG = [re.compile(p) for p in [
    r"\bsupuest\w*\b",        # supuesta/o — "la supuesta justicia social"
    r"\bllamad\w*\b",         # llamada/o  — "la llamada redistribucion"
    r"\bmal\s+llamad\w*\b",   # mal llamada/o
    r"\brechaz\w*\b",         # rechazo/an/amos
    r"\bcritic\w*\b",         # critico/a/amos
    r"\belimin\w*\b",         # eliminar/o — direction-gated below
    r"\ben\s+contra\b",
    r"\bopuest\w*\b",
    r"\ben\s+nombre\s+de\b",
    r"\blo\s+que\s+llaman\b",
    r"\bso\s+pretexto\b",
    r"\bnegar\w*\b",
    r"\bcombat\w*\b",
    r"\bfals\w*\b",
    r"\bideolog\w*\b",
]]

_NEG_WEAK = [re.compile(p) for p in [
    r"\bno\b",
    r"\bnunca\b",
    r"\bjamas\b",
]]


def is_negated(sentence: str, match_start: int, direction: str = "both") -> bool:
    """
    Return True if the match at match_start is preceded by critical-usage or
    negation signals within the configured word windows.

    direction : "hawkish" | "dovish" | "both"
        elimin* is skipped for hawkish terms — Milei uses "eliminar la
        inflacion / el impuesto inflacionario" as hawkish framing, not
        opposition.  Without this gate, ~11 Milei hawkish hits are
        incorrectly suppressed.
    """
    pre_words = sentence[:match_start].split()

    window_strong = " ".join(pre_words[-NEG_WINDOW_STRONG:])
    for pat in _NEG_STRONG:
        if pat.pattern == r"\belimin\w*\b" and direction == "hawkish":
            continue
        if pat.search(window_strong):
            return True

    window_weak = " ".join(pre_words[-NEG_WINDOW_WEAK:])
    return any(p.search(window_weak) for p in _NEG_WEAK)


# ── Text helpers ──────────────────────────────────────────────────────────────
_CLEAN_RE  = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")

def normalise(text: str) -> str:
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = _CLEAN_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()

_SENT_RE = re.compile(r"(?<=[.!?,;:])\s+")

def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts if parts else [text]

def term_to_pattern(term: str) -> re.Pattern:
    """Each word gets \\w* suffix for morphological flexibility."""
    words = term.split()
    return re.compile(r"\s+".join(w + r"\w*" for w in words))

def load_dictionary(path: str) -> list[tuple[str, re.Pattern]]:
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(normalise(line))
    unique_terms = sorted(set(terms), key=len, reverse=True)
    return [(t, term_to_pattern(t)) for t in unique_terms]


# ── Keyword fiscal filter (v7) ────────────────────────────────────────────────

def build_fiscal_patterns() -> list[re.Pattern]:
    """
    Compile regex patterns for the keyword fiscal filter.
    Each keyword gets a \\w* suffix for morphological flexibility,
    consistent with how dictionary terms are matched.
    """
    patterns = []
    for kw in FISCAL_KEYWORDS:
        norm = normalise(kw)
        words = norm.split()
        pat = re.compile(
            r"\b" + r"\s+".join(re.escape(w) + r"\w*" for w in words)
        )
        patterns.append(pat)
    return patterns


_FISCAL_PATTERNS: list[re.Pattern] = []   # populated in run()


def is_fiscal_paragraph(text_norm: str) -> bool:
    """Return True if the normalised paragraph text contains any fiscal keyword."""
    return any(pat.search(text_norm) for pat in _FISCAL_PATTERNS)


# ── Paragraph-level scoring ───────────────────────────────────────────────────

def score_paragraph(
    text: str,
    hawkish_patterns: list[tuple[str, re.Pattern]],
    dovish_patterns:  list[tuple[str, re.Pattern]],
) -> dict:
    """
    Scan a paragraph for hawkish/dovish dictionary hits with negation detection.

    Returns binary flags (has_hawkish, has_dovish) plus raw hit counts and
    matched/negated term lists for audit.  No TF normalisation — each
    paragraph is a single directional vote.
    """
    norm      = normalise(text)
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

    h_hits = sum(h_term_counts.values())
    d_hits = sum(d_term_counts.values())

    return {
        "hawkish_hits":          h_hits,
        "dovish_hits":           d_hits,
        "hawkish_negated":       sum(h_neg_counts.values()),
        "dovish_negated":        sum(d_neg_counts.values()),
        "has_hawkish":           int(h_hits > 0),   # binary vote
        "has_dovish":            int(d_hits > 0),   # binary vote
        "hawkish_terms_matched": list(h_term_counts.keys()),
        "dovish_terms_matched":  list(d_term_counts.keys()),
        "hawkish_terms_negated": list(h_neg_counts.keys()),
        "dovish_terms_negated":  list(d_neg_counts.keys()),
    }


# ── Speech-level aggregation ──────────────────────────────────────────────────

def aggregate_to_speech(para_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate paragraph-level votes to speech level.

    Counts are split by fiscal/non-fiscal paragraphs.  The signal is built
    exclusively from fiscal paragraphs; non-fiscal counts are retained for
    diagnostic comparison only.
    """
    records = []
    for speech_id, grp in para_df.groupby("speech_id"):
        meta    = grp.iloc[0]
        fiscal  = grp[grp["is_fiscal"] == True]
        nfiscal = grp[grp["is_fiscal"] == False]

        P = len(fiscal)
        H = fiscal["has_hawkish"].sum()
        D = fiscal["has_dovish"].sum()

        records.append({
            "speech_id":             speech_id,
            "date":                  meta["date"],
            "president":             meta["president"],
            "president_id":          meta["president_id"],
            "year_month":            meta["year_month"],
            # Paragraph counts
            "n_paragraphs":          len(grp),
            "n_fiscal_paragraphs":   P,
            # Fiscal paragraph votes
            "hawkish_fiscal_paras":  int(H),
            "dovish_fiscal_paras":   int(D),
            "neutral_fiscal_paras":  int(P - (grp["is_fiscal"] & ((grp["has_hawkish"] | grp["has_dovish"]))).sum()),
            # Speech-level net hawkish share (undefined if no fiscal paragraphs)
            "net_hawkish_share":     (H - D) / P if P > 0 else np.nan,
            # Raw hit totals (for audit)
            "hawkish_hits_total":    grp["hawkish_hits"].sum(),
            "dovish_hits_total":     grp["dovish_hits"].sum(),
            "hawkish_negated_total": grp["hawkish_negated"].sum(),
            "dovish_negated_total":  grp["dovish_negated"].sum(),
            # Non-fiscal for diagnostics
            "hawkish_nonfiscal_paras": int(nfiscal["has_hawkish"].sum()),
            "dovish_nonfiscal_paras":  int(nfiscal["has_dovish"].sum()),
        })

    speech_df = pd.DataFrame(records)
    speech_df["date"] = pd.to_datetime(speech_df["date"])
    speech_df.sort_values("date", inplace=True, ignore_index=True)
    return speech_df


# ── Monthly aggregation (BVAR input) ─────────────────────────────────────────

def aggregate_to_monthly(speech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate speech-level paragraph counts to a monthly signal.

    Primary signal (EPU-style):
        signal_t = (H_t − D_t) / P_t

    where H_t, D_t, P_t are summed across all speeches in the month.
    This pools paragraphs from all speeches before dividing, which is more
    stable than averaging speech-level net_hawkish_share (avoids giving
    equal weight to a 1-paragraph speech and a 30-paragraph speech).

    Robustness column:
        signal_rob_t = (H_t − D_t) / N_t
    where N_t = total paragraphs in the month (fiscal + non-fiscal).
    This matches the EPU denominator most closely and penalises months
    where fiscal discourse is a small fraction of total speech volume.

    Both series are z-scored over the full cross-president sample.
    Primary BVAR column: net_hawkish_z
    """
    core = speech_df[speech_df["president"].isin(PRES_ORDER)].copy()

    records = []
    for (ym, pres), grp in core.groupby(["year_month", "president"], observed=True):
        P_t = grp["n_fiscal_paragraphs"].sum()
        H_t = grp["hawkish_fiscal_paras"].sum()
        D_t = grp["dovish_fiscal_paras"].sum()
        N_t = grp["n_paragraphs"].sum()   # total paragraphs (robustness denominator)

        records.append({
            "year_month":      ym,
            "president":       pres,
            "n_speeches":      len(grp),
            "n_fiscal_paras":  int(P_t),
            "n_total_paras":   int(N_t),
            "H_t":             int(H_t),
            "D_t":             int(D_t),
            # Primary: normalise by fiscal paragraphs
            "net_hawkish":     (H_t - D_t) / P_t if P_t > 0 else np.nan,
            # Robustness: normalise by all paragraphs (EPU-style denominator)
            "net_hawkish_rob": (H_t - D_t) / N_t if N_t > 0 else np.nan,
        })

    monthly = pd.DataFrame(records)
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])
    monthly.sort_values(["ym_dt", "president"], inplace=True, ignore_index=True)

    # ── Winsorise at 2.5th / 97.5th percentile before z-scoring ─────────────
    # Thin months (low P_t) can produce extreme (H−D)/P ratios from a handful
    # of hits.  Winsorising the raw signal before z-scoring prevents a single
    # P_t=2 month (e.g. Macri 2017-12: raw=0.50 → z≈+2.4 unwinsorised) from
    # dominating the variance.  This is standard in the EPU / text-based macro
    # literature (Baker–Bloom–Davis 2016 use a similar outlier cap).
    # The winsorised primary series is the BVAR input (net_hawkish_z).
    # The unwinsorised series is retained as net_hawkish_z_raw for audit.
    for raw_col in ["net_hawkish", "net_hawkish_rob"]:
        lo = monthly[raw_col].quantile(0.025)
        hi = monthly[raw_col].quantile(0.975)
        monthly[raw_col + "_wins"] = monthly[raw_col].clip(lo, hi)

    # ── Z-score over full cross-president sample ──────────────────────────────
    for raw_col, z_col in [
        ("net_hawkish_wins",     "net_hawkish_z"),      # PRIMARY — winsorised
        ("net_hawkish_rob_wins", "net_hawkish_rob_z"),  # robustness — winsorised
        ("net_hawkish",          "net_hawkish_z_raw"),  # audit — unwinsorised
    ]:
        mu  = monthly[raw_col].mean()
        sig = monthly[raw_col].std()
        monthly[z_col] = (monthly[raw_col] - mu) / sig if sig > 0 else 0.0

    monthly.drop(columns=["ym_dt"], inplace=True)
    return monthly


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scoring_overview(speech_df: pd.DataFrame, monthly_df: pd.DataFrame):
    core = speech_df[speech_df["president"].isin(PRES_ORDER)].copy()
    monthly_df = monthly_df.copy()
    monthly_df["ym_dt"] = pd.to_datetime(monthly_df["year_month"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    inaug_dates = ["2015-12-10", "2019-12-10", "2023-12-10"]

    # ── Panel 1: monthly net hawkish z-score ──────────────────────────────────
    ax = axes[0]
    for pres in PRES_ORDER:
        sub = monthly_df[monthly_df["president"] == pres].sort_values("ym_dt")
        ax.plot(sub["ym_dt"], sub["net_hawkish_z"],
                color=PRES_COLORS[pres], linewidth=1.8, label=pres)
        ax.fill_between(sub["ym_dt"], sub["net_hawkish_z"], 0,
                        color=PRES_COLORS[pres], alpha=0.12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for d in inaug_dates:
        ax.axvline(pd.Timestamp(d), color="grey", linewidth=1,
                   linestyle="--", alpha=0.5)
    ax.set_title("Monthly net hawkish share — z-score (BVAR input: net_hawkish_z)")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=9)

    # ── Panel 2: H_t and D_t stacked bars by president-month ─────────────────
    ax = axes[1]
    for pres in PRES_ORDER:
        sub = monthly_df[monthly_df["president"] == pres].sort_values("ym_dt")
        ax.bar(sub["ym_dt"], sub["H_t"],
               color=PRES_COLORS[pres], alpha=0.8, width=20, label=f"{pres} H")
        ax.bar(sub["ym_dt"], -sub["D_t"],
               color=PRES_COLORS[pres], alpha=0.35, width=20, label=f"{pres} D")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly hawkish (↑) and dovish (↓) fiscal paragraph counts")
    ax.set_ylabel("Paragraph count")
    handles = [Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER]
    ax.legend(handles=handles, fontsize=9)

    # ── Panel 3: primary vs robustness z-score scatter ────────────────────────
    ax = axes[2]
    for pres in PRES_ORDER:
        sub = monthly_df[monthly_df["president"] == pres].dropna(
            subset=["net_hawkish_z", "net_hawkish_rob_z"])
        ax.scatter(sub["net_hawkish_z"], sub["net_hawkish_rob_z"],
                   color=PRES_COLORS[pres], alpha=0.5, s=20, label=pres)
    lims_x = monthly_df["net_hawkish_z"].agg(["min", "max"])
    lims_y = monthly_df["net_hawkish_rob_z"].agg(["min", "max"])
    lims = [min(lims_x["min"], lims_y["min"]) - 0.1,
            max(lims_x["max"], lims_y["max"]) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Primary z-score (÷ fiscal paragraphs)")
    ax.set_ylabel("Robustness z-score (÷ total paragraphs)")
    ax.set_title("Primary vs robustness signal — should be tightly correlated")
    handles = [Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER]
    ax.legend(handles=handles, fontsize=9)

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

    # ── v7: keyword fiscal filter ─────────────────────────────────────────────
    # Build compiled patterns (populate module-level list used by is_fiscal_paragraph)
    global _FISCAL_PATTERNS
    _FISCAL_PATTERNS = build_fiscal_patterns()
    print(f"  Fiscal keyword filter: {len(FISCAL_KEYWORDS)} keywords")

    # Apply filter on normalised text — symmetric for hawkish and dovish content
    para_df["text_norm"] = para_df["text_para"].apply(
        lambda t: normalise(str(t))
    )
    para_df["is_fiscal"] = para_df["text_norm"].apply(is_fiscal_paragraph)
    # Retain LDA fiscal_topic_prob column for validation exhibit (not used for scoring)

    n_total  = len(para_df)
    n_fiscal = para_df["is_fiscal"].sum()
    print(f"  {n_total:,} paragraphs loaded")
    print(f"  {n_fiscal:,} fiscal by keyword filter ({n_fiscal/n_total*100:.1f}%)")

    # Overlap with LDA filter (validation check)
    if "fiscal_topic_prob" in para_df.columns:
        lda_fiscal = para_df["fiscal_topic_prob"] >= 0.15
        overlap = (para_df["is_fiscal"] & lda_fiscal).sum()
        kw_only = (para_df["is_fiscal"] & ~lda_fiscal).sum()
        lda_only = (~para_df["is_fiscal"] & lda_fiscal).sum()
        print(f"  LDA/keyword overlap: both={overlap:,} | kw-only={kw_only:,} | lda-only={lda_only:,}")

    # ── 3. Score each paragraph (all paragraphs — filter applied at step 4) ──
    print("\nScoring paragraphs (negation-aware, sentence-level)...")
    score_records = []
    for _, row in para_df.iterrows():
        scores = score_paragraph(
            str(row["text_para"]),
            hawkish_patterns,
            dovish_patterns,
        )
        score_records.append(scores)

    score_df    = pd.DataFrame(score_records)
    para_scored = pd.concat(
        [para_df.reset_index(drop=True), score_df.reset_index(drop=True)],
        axis=1,
    )

    # Quick hit-rate diagnostics
    fiscal_scored = para_scored[para_scored["is_fiscal"]]
    any_hit_fiscal = (fiscal_scored["hawkish_hits"] + fiscal_scored["dovish_hits"]) > 0
    print(f"  Fiscal paragraphs with ≥1 accepted hit : "
          f"{any_hit_fiscal.sum():,} / {len(fiscal_scored):,} "
          f"({any_hit_fiscal.mean()*100:.1f}%)")
    print(f"  Hawkish fiscal paras: {fiscal_scored['has_hawkish'].sum():,}")
    print(f"  Dovish  fiscal paras: {fiscal_scored['has_dovish'].sum():,}")

    # ── 4. Aggregate to speech level ──────────────────────────────────────────
    print("\nAggregating to speech level...")
    speech_df = aggregate_to_speech(para_scored)
    print(f"  {len(speech_df):,} speeches")

    # ── 5. Monthly aggregation (BVAR input) ───────────────────────────────────
    print("\nAggregating to monthly (EPU-style paragraph counts)...")
    monthly_df = aggregate_to_monthly(speech_df)
    print(f"  {len(monthly_df)} month-president rows")

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    drop_cols = [
        "hawkish_terms_matched", "dovish_terms_matched",
        "hawkish_terms_negated", "dovish_terms_negated",
        "text_norm",
    ]
    para_out = os.path.join(INTERIM_DIR, "paragraphs_scored.csv")
    para_scored.drop(columns=drop_cols, errors="ignore").to_csv(para_out, index=False)
    print(f"\nSaved: {para_out}")

    speech_out = os.path.join(INTERIM_DIR, "speeches_scored.csv")
    speech_df.to_csv(speech_out, index=False)
    print(f"Saved: {speech_out}")

    monthly_out = os.path.join(INTERIM_DIR, "monthly_signal.csv")
    monthly_df.to_csv(monthly_out, index=False)
    print(f"Saved: {monthly_out}")

    # Clean BVAR-ready output (minimal columns)
    processed_dir = os.path.join(_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    bvar_cols = ["year_month", "president", "net_hawkish_z", "net_hawkish_z_raw",
                 "net_hawkish_rob_z", "H_t", "D_t", "n_fiscal_paras", "n_speeches"]
    bvar_out = os.path.join(processed_dir, "bvar_signal.csv")
    monthly_df[[c for c in bvar_cols if c in monthly_df.columns]].to_csv(bvar_out, index=False)
    print(f"Saved: {bvar_out}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_scoring_overview(speech_df, monthly_df)

    # ── 8. Audit report ───────────────────────────────────────────────────────
    # Re-scan for per-term counts (by president) — used for validation exhibit
    print("\nBuilding term-level audit report...")
    hit_h:  dict[str, int] = {}
    hit_d:  dict[str, int] = {}
    neg_h:  dict[str, int] = {}
    neg_d:  dict[str, int] = {}
    pres_stats: dict[str, dict] = {
        p: {"h_paras": 0, "d_paras": 0, "h_acc": 0, "d_acc": 0,
            "h_neg": 0, "d_neg": 0, "fiscal_paras": 0}
        for p in PRES_ORDER
    }

    for _, row in para_scored.iterrows():
        pres = row["president"]
        if pres not in pres_stats:
            continue
        norm  = normalise(str(row["text_para"]))
        sents = split_sentences(norm)

        for term, pat in hawkish_patterns:
            for sent in sents:
                for m in pat.finditer(sent):
                    if is_negated(sent, m.start(), direction="hawkish"):
                        neg_h[term] = neg_h.get(term, 0) + 1
                        pres_stats[pres]["h_neg"] += 1
                    else:
                        hit_h[term] = hit_h.get(term, 0) + 1
                        pres_stats[pres]["h_acc"] += 1

        for term, pat in dovish_patterns:
            for sent in sents:
                for m in pat.finditer(sent):
                    if is_negated(sent, m.start(), direction="dovish"):
                        neg_d[term] = neg_d.get(term, 0) + 1
                        pres_stats[pres]["d_neg"] += 1
                    else:
                        hit_d[term] = hit_d.get(term, 0) + 1
                        pres_stats[pres]["d_acc"] += 1

        if row["is_fiscal"]:
            pres_stats[pres]["fiscal_paras"] += 1
            pres_stats[pres]["h_paras"] += row["has_hawkish"]
            pres_stats[pres]["d_paras"] += row["has_dovish"]

    # ── 9. Summary text ───────────────────────────────────────────────────────
    lines = [
        "=== DICTIONARY SCORING SUMMARY (v7 — keyword fiscal filter) ===",
        f"Hawkish terms : {len(hawkish_patterns)}",
        f"Dovish terms  : {len(dovish_patterns)}",
        f"Fiscal keywords: {len(FISCAL_KEYWORDS)}",
        "",
        f"Total paragraphs : {n_total:,}",
        f"Fiscal paragraphs: {n_fiscal:,} ({n_fiscal/n_total*100:.1f}%)",
        "",
        "Signal formula: signal_t = (H_t − D_t) / P_t",
        "  H_t = hawkish-hit fiscal paragraphs in month t",
        "  D_t = dovish-hit  fiscal paragraphs in month t",
        "  P_t = total fiscal paragraphs in month t",
        "",
        "── PARAGRAPH VOTE COUNTS BY PRESIDENT ────────────────────────────────",
        f"  {'President':<8} {'Fiscal':>8} {'H_paras':>8} {'D_paras':>8}"
        f"  {'H/P':>8} {'D/P':>8} {'(H-D)/P':>8}",
    ]
    for pres in PRES_ORDER:
        s  = pres_stats[pres]
        P  = s["fiscal_paras"]
        H  = s["h_paras"]
        D  = s["d_paras"]
        hp = H / P if P > 0 else float("nan")
        dp = D / P if P > 0 else float("nan")
        nd = (H - D) / P if P > 0 else float("nan")
        lines.append(
            f"  {pres:<8} {P:>8,} {H:>8,} {D:>8,}"
            f"  {hp:>8.4f} {dp:>8.4f} {nd:>8.4f}"
        )

    lines += [
        "",
        "── MONTHLY SIGNAL (BVAR input: net_hawkish_z) ────────────────────────",
        f"  {'President':<8} {'N months':>9} {'Mean':>9} {'Std':>9}"
        f" {'Min':>9} {'Max':>9} {'Mean-z':>9} {'Std-z':>9}",
        "-" * 78,
    ]
    for pres in PRES_ORDER:
        sub  = monthly_df[monthly_df["president"] == pres]
        raw  = sub["net_hawkish"].dropna()
        z    = sub["net_hawkish_z"].dropna()
        lines.append(
            f"  {pres:<8} {len(raw):>9d} {raw.mean():>9.4f} {raw.std():>9.4f}"
            f" {raw.min():>9.4f} {raw.max():>9.4f}"
            f" {z.mean():>9.4f} {z.std():>9.4f}"
        )

    lines += [
        "",
        "── NEGATION SUPPRESSION REPORT ────────────────────────────────────────",
    ]
    for pres in PRES_ORDER:
        s = pres_stats[pres]
        hg = s["h_acc"] + s["h_neg"]
        dg = s["d_acc"] + s["d_neg"]
        hp = s["h_neg"] / hg * 100 if hg > 0 else 0
        dp = s["d_neg"] / dg * 100 if dg > 0 else 0
        lines.append(
            f"  {pres:<6}  hawkish: {s['h_acc']:4d} accepted, {s['h_neg']:3d} negated "
            f"({hp:.1f}%)  |  dovish: {s['d_acc']:4d} accepted, {s['d_neg']:3d} negated "
            f"({dp:.1f}%)"
        )

    lines += ["", "Top negated dovish terms:"]
    for term, cnt in sorted(neg_d.items(), key=lambda x: x[1], reverse=True)[:12]:
        lines.append(f"  {cnt:5d}  {term}")

    lines += ["", "Top negated hawkish terms:"]
    for term, cnt in sorted(neg_h.items(), key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"  {cnt:5d}  {term}")

    lines += ["", "── TOP ACCEPTED TERMS ─────────────────────────────────────────────",
              "Top 15 hawkish terms (paragraph hits):"]
    for term, cnt in sorted(hit_h.items(), key=lambda x: x[1], reverse=True)[:15]:
        lines.append(f"  {cnt:5d}  {term}")

    lines += ["", "Top 15 dovish terms (paragraph hits):"]
    for term, cnt in sorted(hit_d.items(), key=lambda x: x[1], reverse=True)[:15]:
        lines.append(f"  {cnt:5d}  {term}")

    zero_h = sorted({t for t, _ in hawkish_patterns} - set(hit_h))
    zero_d = sorted({t for t, _ in dovish_patterns}  - set(hit_d))
    lines += ["", f"ZERO-HIT HAWKISH ({len(zero_h)}):"]
    lines += [f"  {t}" for t in zero_h]
    lines += ["", f"ZERO-HIT DOVISH ({len(zero_d)}):"]
    lines += [f"  {t}" for t in zero_d]

    summary = "\n".join(lines)
    print("\n" + summary)

    path = os.path.join(TABLES_DIR, "scoring_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved: {path}")

    return para_scored, speech_df, monthly_df


if __name__ == "__main__":
    para_scored, speech_df, monthly_df = run()
