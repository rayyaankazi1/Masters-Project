"""
signal/scoring/llm_scoring.py  (Stage 5 — LLM cross-validation)
────────────────────────────────────────────────────────────────────────
LLM-based fiscal stance scoring for Argentine presidential speeches.

Motivation
──────────
The dictionary signal (Stage 4) is built from fiscal-accounting vocabulary —
it captures "deficit cero", "superavit primario", "gasto social", etc.  It
misses speeches that communicate fiscal stance through ideological argument
rather than accounting phrases.  The clearest case is Milei's January 2024
Davos speech: the paragraphs about taxes-as-coercion and monetary-emission-
as-control are strongly hawkish in content but contain no dictionary terms
(except subsidios which paradoxically scores dovish).

This script uses the Claude API to score each fiscal paragraph on fiscal
stance directly from content, with no vocabulary constraints.  Crucially:
  • President identity is NEVER shown to the model — no speaker metadata,
    no speech titles, no dates.  The model scores what the paragraph says,
    not who said it.
  • The same rubric is applied to all presidents uniformly.
  • This avoids the circularity problem: we cannot get a self-fulfilling
    Milei signal by including Milei-specific vocabulary, because we never
    tell the model the speaker is Milei.

Methodology
──────────────
Stage 1 — Keyword fiscal filter (inherited from tfidf_dictionary.py)
    Only paragraphs flagged `is_fiscal=True` in paragraphs_scored.csv
    are scored.  This is the same 2,920-paragraph set used by the dictionary.

Stage 2 — LLM scoring (this script)
    For each fiscal paragraph, call the Claude API with a structured rubric.
    Response: {"score": -1|0|1, "reason": "<brief explanation>"}
    Scores: +1 hawkish | 0 neutral | -1 dovish

Stage 3 — Monthly aggregation (EPU-style)
    signal_t = (H_t − D_t) / P_t
    Identical formula to the dictionary signal so results are directly
    comparable.  Z-scored over full cross-president sample.

Stage 4 — Cross-validation
    Compute Pearson and Spearman correlation between LLM and dictionary
    monthly z-scores.  Target: ρ > 0.65 (CLAUDE.md Stage 5 target).
    Log per-month divergences for audit.

Checkpoint / resume
──────────────────
Scores are checkpointed to data/interim/llm_scores_checkpoint.json after
each batch.  If the script is interrupted, re-running it will skip already-
scored paragraphs.  Delete the checkpoint file to rescore from scratch.

Usage
──────
    export ANTHROPIC_API_KEY="sk-ant-..."
    cd ~/Desktop/Masters-Project
    source .venv/bin/activate

    # Dry run — show cost estimate, do not call API
    python signal/scoring/llm_scoring.py --dry-run

    # Full run (default model: claude-haiku-4-5-20251001)
    python signal/scoring/llm_scoring.py

    # Use a more capable model for higher accuracy
    python signal/scoring/llm_scoring.py --model claude-sonnet-4-6

    # Score a random sample first (useful for testing)
    python signal/scoring/llm_scoring.py --sample 200

    # Override batch size (default: 10 paragraphs per API call)
    python signal/scoring/llm_scoring.py --batch-size 5

Reads
──────
    data/interim/paragraphs_scored.csv      (produced by tfidf_dictionary.py)
    data/interim/monthly_signal.csv         (dictionary signal for comparison)
    data/interim/llm_scores_checkpoint.json (if resuming)

Writes
──────
    data/interim/llm_scores_checkpoint.json     individual paragraph scores
    data/interim/paragraphs_llm_scored.csv      paragraphs with LLM scores
    data/interim/monthly_signal_llm.csv         monthly LLM signal
    data/processed/bvar_signal_llm.csv          clean BVAR-ready LLM signal
    outputs/figures/llm_vs_dict_signal.png      validation chart
    outputs/tables/llm_scoring_summary.txt      audit report
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not found. Run: pip install anthropic")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_SCORED   = os.path.join(_ROOT, "data", "interim", "paragraphs_scored.csv")
MONTHLY_DICT  = os.path.join(_ROOT, "data", "interim", "monthly_signal.csv")
CHECKPOINT    = os.path.join(_ROOT, "data", "interim", "llm_scores_checkpoint.json")
INTERIM_DIR   = os.path.join(_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
FIGURES_DIR   = os.path.join(_ROOT, "outputs", "figures")
TABLES_DIR    = os.path.join(_ROOT, "outputs", "tables")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "claude-haiku-4-5-20251001"
DEFAULT_BATCH_SIZE = 10
PRES_ORDER         = ["Macri", "AF", "Milei"]
PRES_COLORS        = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}

# Approximate token costs (USD per million tokens) — update if pricing changes
COST_PER_M_IN  = {"claude-haiku-4-5-20251001": 0.80,  "claude-sonnet-4-6": 3.00}
COST_PER_M_OUT = {"claude-haiku-4-5-20251001": 4.00,  "claude-sonnet-4-6": 15.00}

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a fiscal policy analyst. Your task is to score excerpts from political speeches for their fiscal policy direction.

Scoring rubric:
+1 (HAWKISH) — The paragraph supports or signals fiscal TIGHTENING. Examples:
  • Deficit reduction, balanced budget, surplus targets
  • Spending cuts, austerity, reducing public expenditure
  • Anti-inflation commitment, monetary restraint
  • Tax reform to reduce burden (fewer, lower taxes)
  • Privatisation, deregulation, market liberalisation
  • Criticising state spending, subsidies, or debt as harmful/unsustainable
  • Arguing that government intervention distorts prices or reduces growth

0 (NEUTRAL) — The paragraph describes fiscal policy without a clear directional signal, discusses both sides, or is primarily narrative/contextual without policy endorsement.

-1 (DOVISH) — The paragraph supports or signals fiscal EXPANSION. Examples:
  • Increased public spending, social transfers, welfare expansion
  • Public investment programmes (infrastructure, education, health)
  • Debt accommodation, debt restructuring, borrowing to fund growth
  • Economic stimulus, counter-cyclical spending
  • Protection of subsidies, social programmes, state enterprises

Critical rules:
1. Score the CONTENT and INTENT of the paragraph, not its vocabulary.
2. A paragraph CRITICISING subsidies/state spending scores +1 (hawkish), not -1.
3. A paragraph describing what opponents believe (hypothetical/ironic framing) scores 0.
4. The speaker's identity is unknown to you — do not assume who is speaking.
5. Texts are in Spanish. Score from the meaning, not surface words.
6. Respond ONLY with valid JSON — no explanations outside the JSON."""

USER_PROMPT_TEMPLATE = """Score each of the following {n} paragraphs from a political speech.

Respond with ONLY a JSON array of {n} objects, one per paragraph IN ORDER:
[{{"id": <integer>, "score": <-1|0|1>, "reason": "<8–12 words summarising the fiscal direction>"}}]

Paragraphs:
{paragraphs}"""


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip accents, remove punctuation — matches tfidf_dictionary.py."""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_paragraph(text: str, max_chars: int = 800) -> str:
    """Truncate very long paragraphs to control token costs."""
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if last_stop > max_chars // 2:
        return cut[:last_stop + 1].strip() + " [...]"
    return cut.strip() + " [...]"


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(n_paragraphs: int, batch_size: int, model: str) -> dict:
    """
    Estimate API cost before running.

    Assumptions:
      • Average paragraph: 120 Spanish words ≈ 160 tokens
      • System prompt: ~350 tokens (shared across the request)
      • Per-paragraph overhead in prompt: ~30 tokens (numbering, formatting)
      • Output per paragraph: ~25 tokens (JSON score + reason)
    """
    n_batches        = max(1, n_paragraphs // batch_size)
    tokens_sys       = 350                          # system prompt (per call)
    tokens_para_in   = (160 + 30) * batch_size      # paragraph text + formatting
    tokens_in_total  = n_batches * (tokens_sys + tokens_para_in)
    tokens_out_total = n_paragraphs * 25

    cost_in  = tokens_in_total  / 1e6 * COST_PER_M_IN.get(model, 1.00)
    cost_out = tokens_out_total / 1e6 * COST_PER_M_OUT.get(model, 5.00)

    return {
        "n_paragraphs": n_paragraphs,
        "n_batches":    n_batches,
        "tokens_in":    tokens_in_total,
        "tokens_out":   tokens_out_total,
        "cost_usd":     cost_in + cost_out,
    }


# ── API scoring ───────────────────────────────────────────────────────────────

def parse_score_response(raw: str, expected_ids: list[int]) -> dict[int, dict]:
    """
    Parse the JSON array returned by the API.

    Returns a dict mapping para_id → {"score": int, "reason": str}.
    Falls back gracefully if the model returns malformed JSON.
    """
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    # Sometimes the model wraps in a single object — try array first
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract the JSON array with regex
        m = re.search(r"\[.*\]", clean, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, list):
        data = [data]

    result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid   = item.get("id")
        score = item.get("score")
        if pid is None or score is None:
            continue
        try:
            pid   = int(pid)
            score = int(score)
        except (TypeError, ValueError):
            continue
        if score not in (-1, 0, 1):
            score = 0   # default to neutral if out of range
        result[pid] = {
            "llm_score":  score,
            "llm_reason": str(item.get("reason", ""))[:120],
        }
    return result


def score_batch(
    client: "anthropic.Anthropic",
    batch: list[dict],
    model: str,
    retry_limit: int = 3,
) -> dict[int, dict]:
    """
    Score a batch of paragraphs via the Claude API.

    Each item in `batch` must have keys: para_id, text_para.
    Returns a dict mapping para_id → {"llm_score": int, "llm_reason": str}.
    """
    para_lines = "\n".join(
        f"{i+1}. [id={row['para_id']}] {truncate_paragraph(str(row['text_para']))}"
        for i, row in enumerate(batch)
    )
    user_msg = USER_PROMPT_TEMPLATE.format(
        n=len(batch),
        paragraphs=para_lines,
    )

    for attempt in range(retry_limit):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text
            parsed = parse_score_response(raw, [r["para_id"] for r in batch])

            # Check we got scores for all expected paragraphs
            expected = {r["para_id"] for r in batch}
            missing  = expected - set(parsed.keys())
            if missing:
                # Partial result — retry
                if attempt < retry_limit - 1:
                    time.sleep(2 ** attempt)
                    continue
                # Last attempt: fill missing with neutral
                for pid in missing:
                    parsed[pid] = {"llm_score": 0, "llm_reason": "parse_fallback"}

            return parsed

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [Rate limit] waiting {wait}s...", end="", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"\n  [API error attempt {attempt+1}] {e}")
            if attempt < retry_limit - 1:
                time.sleep(5 * (attempt + 1))
            else:
                # Return neutral for all paragraphs on persistent failure
                return {r["para_id"]: {"llm_score": 0, "llm_reason": "api_error"} for r in batch}

    return {r["para_id"]: {"llm_score": 0, "llm_reason": "retry_exhausted"} for r in batch}


# ── Monthly aggregation ───────────────────────────────────────────────────────

def aggregate_monthly(para_df: pd.DataFrame, score_col: str = "llm_score") -> pd.DataFrame:
    """
    Aggregate paragraph LLM scores to monthly EPU-style signal.

    signal_t = (H_t − D_t) / P_t
    where H_t = paragraphs with score=+1, D_t = paragraphs with score=-1,
    P_t = total fiscal paragraphs in month t (all scores).

    Returns a DataFrame with one row per president-month, z-scored over the
    full cross-president sample.  Identical aggregation logic to
    tfidf_dictionary.py so results are directly comparable.
    """
    fiscal = para_df[para_df["is_fiscal"] == True].copy()

    records = []
    for (ym, pres), grp in fiscal.groupby(["year_month", "president"], observed=True):
        P_t = len(grp)
        H_t = (grp[score_col] == 1).sum()
        D_t = (grp[score_col] == -1).sum()
        records.append({
            "year_month":   ym,
            "president":    pres,
            "n_speeches":   grp["speech_id"].nunique(),
            "n_fiscal_paras": int(P_t),
            "H_t_llm":      int(H_t),
            "D_t_llm":      int(D_t),
            "net_hawkish_llm": (H_t - D_t) / P_t if P_t > 0 else np.nan,
        })

    monthly = pd.DataFrame(records)
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])
    monthly.sort_values(["ym_dt", "president"], inplace=True, ignore_index=True)

    # Winsorise at 2.5/97.5 pct (same as dictionary signal)
    lo = monthly["net_hawkish_llm"].quantile(0.025)
    hi = monthly["net_hawkish_llm"].quantile(0.975)
    monthly["net_hawkish_llm_wins"] = monthly["net_hawkish_llm"].clip(lo, hi)

    # Z-score over full cross-president sample
    mu  = monthly["net_hawkish_llm_wins"].mean()
    sig = monthly["net_hawkish_llm_wins"].std()
    monthly["net_hawkish_llm_z"] = (
        (monthly["net_hawkish_llm_wins"] - mu) / sig if sig > 0 else 0.0
    )
    # Retain unwinsorised z for audit
    mu_r  = monthly["net_hawkish_llm"].mean()
    sig_r = monthly["net_hawkish_llm"].std()
    monthly["net_hawkish_llm_z_raw"] = (
        (monthly["net_hawkish_llm"] - mu_r) / sig_r if sig_r > 0 else 0.0
    )

    monthly.drop(columns=["ym_dt"], inplace=True)
    return monthly[monthly["president"].isin(PRES_ORDER)].copy()


# ── Cross-validation plots ────────────────────────────────────────────────────

def plot_validation(monthly_llm: pd.DataFrame, monthly_dict: pd.DataFrame):
    """
    Four-panel validation chart:
      1. Time series overlay: LLM vs dictionary z-scores
      2. Scatter: monthly LLM vs dictionary (by president)
      3. Per-president mean comparison bar chart
      4. Rolling 6-month correlation
    """
    # Merge on year_month + president
    merged = monthly_llm.merge(
        monthly_dict[["year_month", "president", "net_hawkish_z"]],
        on=["year_month", "president"],
        how="inner",
    )
    merged["ym_dt"] = pd.to_datetime(merged["year_month"])
    merged.sort_values("ym_dt", inplace=True, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    inaug = ["2015-12-10", "2019-12-10", "2023-12-10"]

    # ── 1. Time series overlay ────────────────────────────────────────────────
    ax = axes[0, 0]
    for pres in PRES_ORDER:
        sub = merged[merged["president"] == pres].sort_values("ym_dt")
        ax.plot(sub["ym_dt"], sub["net_hawkish_llm_z"],
                color=PRES_COLORS[pres], linewidth=2.0, label=f"{pres} LLM")
        ax.plot(sub["ym_dt"], sub["net_hawkish_z"],
                color=PRES_COLORS[pres], linewidth=1.2, linestyle="--",
                alpha=0.6, label=f"{pres} Dict")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    for d in inaug:
        ax.axvline(pd.Timestamp(d), color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("LLM signal (solid) vs Dictionary signal (dashed)")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=7, ncol=2)

    # ── 2. Scatter ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    all_valid = merged.dropna(subset=["net_hawkish_llm_z", "net_hawkish_z"])
    r_pearson = all_valid["net_hawkish_llm_z"].corr(all_valid["net_hawkish_z"])
    r_spearman = all_valid["net_hawkish_llm_z"].corr(
        all_valid["net_hawkish_z"], method="spearman"
    )
    for pres in PRES_ORDER:
        sub = all_valid[all_valid["president"] == pres]
        ax.scatter(sub["net_hawkish_z"], sub["net_hawkish_llm_z"],
                   color=PRES_COLORS[pres], alpha=0.55, s=22, label=pres)
    lo = min(all_valid[["net_hawkish_z", "net_hawkish_llm_z"]].min())
    hi = max(all_valid[["net_hawkish_z", "net_hawkish_llm_z"]].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Dictionary z-score")
    ax.set_ylabel("LLM z-score")
    ax.set_title(
        f"Monthly scatter  r={r_pearson:.3f} (Pearson)  ρ={r_spearman:.3f} (Spearman)"
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER], fontsize=9)

    # ── 3. President mean comparison ──────────────────────────────────────────
    ax = axes[1, 0]
    x      = np.arange(len(PRES_ORDER))
    width  = 0.35
    means_llm  = [merged[merged["president"] == p]["net_hawkish_llm_z"].mean() for p in PRES_ORDER]
    means_dict = [merged[merged["president"] == p]["net_hawkish_z"].mean()     for p in PRES_ORDER]
    bars1 = ax.bar(x - width/2, means_llm,  width, label="LLM",        color="#9C27B0", alpha=0.8)
    bars2 = ax.bar(x + width/2, means_dict, width, label="Dictionary", color="#607D8B", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(PRES_ORDER)
    ax.set_title("Mean z-score by president: LLM vs Dictionary")
    ax.set_ylabel("Mean z-score")
    ax.legend(fontsize=9)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # ── 4. Rolling 6m correlation (full panel) ────────────────────────────────
    # Pool all presidents, sort by date, compute rolling Pearson using
    # a manual window to avoid the duplicate-index unstack problem.
    ax = axes[1, 1]
    full = merged.sort_values("ym_dt").dropna(subset=["net_hawkish_llm_z", "net_hawkish_z"]).reset_index(drop=True)
    roll_corr_vals = []
    roll_corr_idx  = []
    window = 6
    for i in range(window - 1, len(full)):
        w = full.iloc[i - window + 1 : i + 1]
        if w[["net_hawkish_llm_z", "net_hawkish_z"]].std().min() == 0:
            continue
        r = w["net_hawkish_llm_z"].corr(w["net_hawkish_z"])
        roll_corr_vals.append(r)
        roll_corr_idx.append(full.iloc[i]["ym_dt"])
    roll_corr = pd.Series(roll_corr_vals, index=roll_corr_idx).dropna()
    ax.plot(roll_corr.index, roll_corr.values, color="#E91E63", linewidth=1.8)
    ax.axhline(0.65, color="green",  linewidth=1, linestyle="--", alpha=0.7, label="Target ρ=0.65")
    ax.axhline(0,    color="black",  linewidth=0.7, linestyle="--", alpha=0.4)
    ax.fill_between(roll_corr.index, roll_corr.values, 0,
                    where=(roll_corr.values >= 0), alpha=0.15, color="#E91E63")
    for d in inaug:
        ax.axvline(pd.Timestamp(d), color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Rolling 6-month Pearson correlation: LLM vs Dictionary")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "llm_vs_dict_signal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return r_pearson, r_spearman


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample: int = None,
        dry_run: bool = False):

    for d in [INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load fiscal paragraphs ─────────────────────────────────────────────
    print(f"Loading {PARA_SCORED}...")
    if not os.path.exists(PARA_SCORED):
        raise FileNotFoundError(
            "paragraphs_scored.csv not found. Run signal/scoring/tfidf_dictionary.py first."
        )
    para_df = pd.read_csv(PARA_SCORED)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)].copy()
    fiscal  = para_df[para_df["is_fiscal"] == True].copy()

    print(f"  Total paragraphs : {len(para_df):,}")
    print(f"  Fiscal paragraphs: {len(fiscal):,}")
    print(f"  By president     : {dict(fiscal['president'].value_counts())}")

    if sample:
        fiscal = fiscal.sample(n=min(sample, len(fiscal)), random_state=42)
        print(f"  Sampled          : {len(fiscal):,} paragraphs")

    # ── 2. Dry run — cost estimate ────────────────────────────────────────────
    est = estimate_cost(len(fiscal), batch_size, model)
    print(f"\n── Cost estimate ({model}) ─────────────────────────────────────")
    print(f"  Paragraphs  : {est['n_paragraphs']:,}")
    print(f"  API calls   : ~{est['n_batches']:,}  (batch size {batch_size})")
    print(f"  Input tokens: ~{est['tokens_in']:,}")
    print(f"  Output tokens: ~{est['tokens_out']:,}")
    print(f"  Estimated cost: ~${est['cost_usd']:.2f} USD")
    print(f"────────────────────────────────────────────────────────────────")

    if dry_run:
        print("\nDry run complete. Use without --dry-run to execute.")
        return

    # ── 3. Load checkpoint ────────────────────────────────────────────────────
    checkpoint: dict[str, dict] = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            checkpoint = json.load(f)
        already_done = sum(1 for k in checkpoint if checkpoint[k].get("llm_score") is not None)
        print(f"\nResuming from checkpoint: {already_done:,} paragraphs already scored")

    # ── 4. Prepare API client ─────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # ── 5. Score paragraphs in batches ────────────────────────────────────────
    rows        = fiscal.to_dict("records")
    to_score    = [r for r in rows if str(r["para_id"]) not in checkpoint]
    n_to_score  = len(to_score)
    n_done      = len(rows) - n_to_score
    total       = len(rows)

    print(f"\nScoring {n_to_score:,} paragraphs (skipping {n_done:,} already done)...")
    print(f"Model: {model}  |  Batch size: {batch_size}")
    print()

    batches = [to_score[i:i+batch_size] for i in range(0, len(to_score), batch_size)]
    start   = time.time()

    for i, batch in enumerate(batches):
        scores = score_batch(client, batch, model)

        for row in batch:
            pid = row["para_id"]
            checkpoint[str(pid)] = scores.get(pid, {"llm_score": 0, "llm_reason": "no_response"})

        # Save checkpoint after every batch
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)

        # Progress display
        done_so_far = n_done + (i + 1) * batch_size
        pct         = min(100, done_so_far / total * 100)
        elapsed     = time.time() - start
        eta         = (elapsed / max(i+1, 1)) * (len(batches) - i - 1)
        print(
            f"  Batch {i+1:4d}/{len(batches):4d}  "
            f"[{pct:5.1f}%]  "
            f"elapsed {elapsed/60:.1f}m  "
            f"ETA {eta/60:.1f}m",
            end="\r", flush=True,
        )

    print(f"\nScoring complete. {len(checkpoint):,} paragraphs scored.")

    # ── 6. Merge LLM scores back into para_df ─────────────────────────────────
    fiscal["llm_score"]  = fiscal["para_id"].apply(
        lambda pid: checkpoint.get(str(pid), {}).get("llm_score", 0)
    )
    fiscal["llm_reason"] = fiscal["para_id"].apply(
        lambda pid: checkpoint.get(str(pid), {}).get("llm_reason", "")
    )

    # ── 7. Monthly aggregation ─────────────────────────────────────────────────
    print("\nAggregating to monthly (EPU-style)...")
    monthly_llm = aggregate_monthly(fiscal, score_col="llm_score")
    print(f"  {len(monthly_llm)} month-president rows")

    # ── 8. Per-president means ────────────────────────────────────────────────
    print("\n── LLM signal means by president ────────────────────────────────")
    for pres in PRES_ORDER:
        sub  = monthly_llm[monthly_llm["president"] == pres]
        raw  = sub["net_hawkish_llm"].dropna()
        z    = sub["net_hawkish_llm_z"].dropna()
        print(f"  {pres:<6}  n={len(raw):3d}  mean={raw.mean():+.3f}  z={z.mean():+.3f}  std-z={z.std():.3f}")

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    # Paragraph-level
    para_out = os.path.join(INTERIM_DIR, "paragraphs_llm_scored.csv")
    fiscal.to_csv(para_out, index=False)
    print(f"\nSaved: {para_out}")

    # Monthly LLM signal
    monthly_out = os.path.join(INTERIM_DIR, "monthly_signal_llm.csv")
    monthly_llm.to_csv(monthly_out, index=False)
    print(f"Saved: {monthly_out}")

    # Clean BVAR-ready output — merges LLM + dictionary signal in one file
    if os.path.exists(MONTHLY_DICT):
        dict_sig = pd.read_csv(MONTHLY_DICT)[
            ["year_month", "president", "net_hawkish_z", "net_hawkish_z_raw",
             "net_hawkish_rob_z", "H_t", "D_t", "n_fiscal_paras", "n_speeches"]
        ]
        bvar = monthly_llm.merge(dict_sig, on=["year_month", "president"], how="left",
                                 suffixes=("_llm", "_dict"))
        bvar_cols = [
            "year_month", "president",
            "net_hawkish_llm_z", "net_hawkish_z",          # primary + LLM
            "net_hawkish_llm_z_raw", "net_hawkish_z_raw",  # unwinsorised audit
            "H_t_llm", "D_t_llm", "H_t", "D_t",           # raw counts
            "n_fiscal_paras", "n_speeches",
        ]
        bvar_out = os.path.join(PROCESSED_DIR, "bvar_signal_llm.csv")
        bvar[[c for c in bvar_cols if c in bvar.columns]].to_csv(bvar_out, index=False)
        print(f"Saved: {bvar_out}")

    # ── 10. Cross-validation against dictionary ───────────────────────────────
    r_pearson, r_spearman = None, None
    if os.path.exists(MONTHLY_DICT):
        print("\nCross-validating against dictionary signal...")
        monthly_dict = pd.read_csv(MONTHLY_DICT)
        r_pearson, r_spearman = plot_validation(monthly_llm, monthly_dict)
        print(f"  Pearson  r  = {r_pearson:.3f}")
        print(f"  Spearman ρ  = {r_spearman:.3f}")
        if r_spearman >= 0.65:
            print(f"  ✓ Target ρ ≥ 0.65 MET")
        else:
            print(f"  ✗ Target ρ ≥ 0.65 NOT met — review dictionary or LLM prompt")

    # ── 11. Audit report ──────────────────────────────────────────────────────
    # Score distribution by president
    lines = [
        "=== LLM SCORING SUMMARY (Stage 5) ===",
        f"Model   : {model}",
        f"Scored  : {len(fiscal):,} fiscal paragraphs",
        f"Batches : {len(batches):,}  (batch size {batch_size})",
        "",
        "── LLM SCORE DISTRIBUTION BY PRESIDENT ──────────────────────────────",
        f"  {'President':<8} {'N':>5}  {'Hawkish':>8}  {'Neutral':>8}  {'Dovish':>8}  "
        f"{'H%':>6}  {'D%':>6}",
    ]
    for pres in PRES_ORDER:
        sub = fiscal[fiscal["president"] == pres]
        n   = len(sub)
        h   = (sub["llm_score"] == 1).sum()
        neu = (sub["llm_score"] == 0).sum()
        d   = (sub["llm_score"] == -1).sum()
        lines.append(
            f"  {pres:<8} {n:>5}  {h:>8}  {neu:>8}  {d:>8}  "
            f"{h/n*100:>5.1f}%  {d/n*100:>5.1f}%"
        )

    lines += [
        "",
        "── MONTHLY SIGNAL SUMMARY (LLM) ──────────────────────────────────────",
        f"  {'President':<8} {'N':>5}  {'Mean-z':>8}  {'Std-z':>8}  {'Min-z':>8}  {'Max-z':>8}",
    ]
    for pres in PRES_ORDER:
        sub = monthly_llm[monthly_llm["president"] == pres]
        z   = sub["net_hawkish_llm_z"].dropna()
        lines.append(
            f"  {pres:<8} {len(z):>5}  {z.mean():>+8.3f}  {z.std():>8.3f}  "
            f"{z.min():>+8.3f}  {z.max():>+8.3f}"
        )

    if r_pearson is not None:
        lines += [
            "",
            "── CROSS-VALIDATION VS DICTIONARY ────────────────────────────────────",
            f"  Pearson  r  = {r_pearson:.4f}",
            f"  Spearman ρ  = {r_spearman:.4f}",
            f"  Target ρ ≥ 0.65: {'MET' if r_spearman >= 0.65 else 'NOT MET'}",
        ]

    # Top divergences (months where LLM and dictionary disagree most)
    if os.path.exists(MONTHLY_DICT):
        monthly_dict = pd.read_csv(MONTHLY_DICT)
        merged = monthly_llm.merge(
            monthly_dict[["year_month", "president", "net_hawkish_z"]],
            on=["year_month", "president"], how="inner"
        ).dropna(subset=["net_hawkish_llm_z", "net_hawkish_z"])
        merged["divergence"] = (merged["net_hawkish_llm_z"] - merged["net_hawkish_z"]).abs()
        top_div = merged.nlargest(15, "divergence")[
            ["year_month", "president", "net_hawkish_llm_z", "net_hawkish_z", "divergence"]
        ]
        lines += ["", "── TOP 15 DIVERGENCES (LLM vs Dictionary) ──────────────────────────"]
        lines.append(f"  {'Month':<10}  {'Pres':<6}  {'LLM-z':>8}  {'Dict-z':>8}  {'|diff|':>8}")
        for _, row in top_div.iterrows():
            lines.append(
                f"  {row['year_month']:<10}  {row['president']:<6}  "
                f"{row['net_hawkish_llm_z']:>+8.3f}  {row['net_hawkish_z']:>+8.3f}  "
                f"{row['divergence']:>8.3f}"
            )

    summary = "\n".join(lines)
    print("\n" + summary)

    path = os.path.join(TABLES_DIR, "llm_scoring_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved: {path}")

    return fiscal, monthly_llm


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5: LLM paragraph scoring for fiscal hawkishness signal."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Paragraphs per API call (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Score a random sample of N paragraphs (omit to score all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate only — do not call the API"
    )
    args = parser.parse_args()

    run(
        model=args.model,
        batch_size=args.batch_size,
        sample=args.sample,
        dry_run=args.dry_run,
    )
