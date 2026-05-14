"""
signal/scoring/llm_scoring_fewshot.py  (Stage 5b — few-shot robustness check)
────────────────────────────────────────────────────────────────────────────────
Few-shot robustness run for the LLM fiscal hawkishness signal.

Purpose
────────
Replicates the zero-shot scoring pipeline (llm_scoring.py) with 9 labeled
calibration examples appended to the system prompt (3 hawkish, 3 neutral,
3 dovish).  Outputs go to separate *_fewshot files so zero-shot results are
never overwritten.  At the end, automatically computes Spearman ρ between
the few-shot and zero-shot monthly series — the key robustness statistic.

Design choices
───────────────
• Examples drawn from the actual corpus (para IDs below); president names
  and dates stripped to preserve blind-scoring.
• The hawkish set includes one ideological example (para 4633, Davos Milei)
  to anchor the model on anti-tax / anti-emission framing that fiscal-
  accounting vocabulary misses.
• The neutral set uses genuinely ambiguous or descriptive paragraphs (IMF
  announcement, pandemic retrospective, energy-sector structural description).
• The dovish set covers three distinct archetypes: obra pública as employment
  generator, social-transfer expansion (AUH/PUAM), and public housing.

Para IDs used as examples (excluded from scoring to avoid circularity):
  Hawkish : 4633 (Milei 2024-01, Davos), 3312 (Milei 2024-09), 3638 (Milei 2024-08)
  Neutral : 7980 (AF 2022-01), 6736 (AF 2022-10), 11561 (Macri 2019-06)
  Dovish  : 8454 (AF 2021-09), 14816 (Macri 2016-10), 6362 (AF 2023-01)

Stability target
─────────────────
Spearman ρ (few-shot vs zero-shot monthly series) > 0.85.
Literature basis: Hansen & Kazinnik (2024) show few-shot examples improve
accuracy on central bank text; Bank of England (2025) SWP 1127 shows ~20–30
examples suffice.  We expect ρ > 0.85 if zero-shot results are stable.

Cost estimates (3,904 paragraphs)
───────────────────────────────────
  Haiku  (claude-haiku-4-5-20251001) : ~$1.41  (~28 min)
  Sonnet (claude-sonnet-4-6)         : ~$5.20  (~35 min)

Usage
──────
    cd ~/Desktop/Masters-Project
    source .venv/bin/activate

    # Dry run — show cost estimate only
    ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring_fewshot.py --dry-run

    # Full run (Haiku, ~$1.41)
    ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring_fewshot.py

    # Sonnet for model-version robustness (~$5.20)
    ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring_fewshot.py --model claude-sonnet-4-6

    # Resume interrupted run (checkpoint is preserved automatically)
    ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring_fewshot.py

    # Start fresh (delete checkpoint first)
    rm data/interim/llm_scores_fewshot_checkpoint.json
    ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring_fewshot.py

Reads
──────
    data/interim/paragraphs_scored.csv          (fiscal paragraph corpus)
    data/interim/monthly_signal.csv             (dictionary signal — for cross-val)
    data/interim/monthly_signal_llm.csv         (zero-shot signal — for comparison)
    data/interim/llm_scores_fewshot_checkpoint.json  (if resuming)

Writes
──────
    data/interim/llm_scores_fewshot_checkpoint.json
    data/interim/paragraphs_llm_scored_fewshot.csv
    data/interim/monthly_signal_llm_fewshot.csv
    data/processed/bvar_signal_llm_fewshot.csv
    outputs/figures/llm_vs_dict_signal_fewshot.png
    outputs/tables/llm_scoring_summary_fewshot.txt
"""

import argparse
import json
import os
import re
import sys
import time

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
MONTHLY_ZS    = os.path.join(_ROOT, "data", "interim", "monthly_signal_llm.csv")
CHECKPOINT    = os.path.join(_ROOT, "data", "interim", "llm_scores_fewshot_checkpoint.json")
INTERIM_DIR   = os.path.join(_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
FIGURES_DIR   = os.path.join(_ROOT, "outputs", "figures")
TABLES_DIR    = os.path.join(_ROOT, "outputs", "tables")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "claude-haiku-4-5-20251001"
DEFAULT_BATCH_SIZE = 10
PRES_ORDER         = ["Macri", "AF", "Milei"]
PRES_COLORS        = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}
COST_PER_M_IN      = {"claude-haiku-4-5-20251001": 0.80, "claude-sonnet-4-6": 3.00}
COST_PER_M_OUT     = {"claude-haiku-4-5-20251001": 4.00, "claude-sonnet-4-6": 15.00}

# Para IDs used as few-shot examples — excluded from scoring to avoid circularity
FEW_SHOT_PARA_IDS = {4633, 3312, 3638, 7980, 6736, 11561, 8454, 14816, 6362}

# ── Prompts ───────────────────────────────────────────────────────────────────
# Base rubric is identical to llm_scoring.py (zero-shot).  The few-shot section
# is appended below — this is the only difference from the zero-shot pipeline.
_BASE_RUBRIC = """You are a fiscal policy analyst. Your task is to score excerpts from political speeches for their fiscal policy direction.

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

# 9 calibration examples — 3 per class, drawn from actual corpus.
# Hawkish examples cover (1) ideological anti-tax framing (Davos-type content
# not caught by fiscal-accounting vocabulary), (2) explicit fiscal balance
# commitment, (3) zero-emission / deficit-zero commitment.
# Neutral examples cover (1) IMF agreement announcement (descriptive, no
# directional endorsement), (2) pandemic retrospective (context, no new signal),
# (3) energy-sector structural description (no fiscal stance).
# Dovish examples cover (1) obra pública as employment generator,
# (2) social-transfer expansion (AUH, PUAM, tarifa social), (3) public housing.
_FEW_SHOT_EXAMPLES = """

Calibration examples (use these to anchor your scoring; do not reference them in reasons):

HAWKISH (+1) examples:
1. "El Estado se financia, a través de impuestos y los impuestos se cobran de manera coactiva. ¿Acaso alguno de nosotros puede decir que pagan los impuestos de manera voluntaria? Lo cual significa que el Estado se financia, a través de la coacción y a mayor carga impositiva mayor es la coacción, menor es la libertad."
   → +1: Argues that taxes are coercive and higher tax burden reduces freedom; endorses smaller government.

2. "Nuestro compromiso con el equilibrio fiscal es inquebrantable y no estamos dispuestos a negociar - bajo ningún punto de vista - el equilibrio fiscal. Ese equilibrio fiscal nos permitió no solo salvar el programa con el Fondo Monetario Internacional, sino que, al mismo tiempo, nos permitió ir armando una curva de pesos."
   → +1: Explicit commitment to fiscal balance as non-negotiable policy anchor; deficit reduction.

3. "Nuestro compromiso férreo, con una política de emisión cero, lo que va a hacer es que dejemos de padecer la inflación... cuando nosotros decimos déficit cero, significa que la deuda no sube. Y si la deuda no sube, la relación deuda-producto... el país es solvente intertemporalmente."
   → +1: Zero monetary emission and deficit-zero commitment; fiscal tightening justified on solvency grounds.

NEUTRAL (0) examples:
4. "Quiero anunciarles que el Gobierno de la Argentina llegó a un acuerdo con el Fondo Monetario Internacional. Gobernar es un ejercicio de responsabilidad, sufríamos un problema y ahora tenemos una solución. Teníamos una soga al cuello, una soga de Damocles, y ahora tenemos un camino que podemos recorrer."
   → 0: Announces IMF agreement as a resolved situation; descriptive of circumstances, no directional fiscal endorsement.

5. "No perdamos nunca de vista que estamos transitando el peor tiempo de la historia... Cuando el sector privado necesitaba auxilio, el Estado fue en su socorro con créditos, fue en su socorro con el ATP... Nosotros hemos podido superar ese tiempo."
   → 0: Describes past emergency pandemic measures in context; retrospective narrative, no forward fiscal direction.

6. "Tenemos que tener la humildad de entender que estamos dentro de un sistema competitivo... lo que estamos logrando en Vaca Muerta... este trabajo en equipo entre el capital y el trabajo, donde todos se tienen que comprometer y todos tienen que cumplir."
   → 0: Discusses energy sector competitiveness and public-private collaboration; no fiscal stance signal.

DOVISH (-1) examples:
7. "Con la obra pública lo que hacemos también es generar muchos puestos de trabajo y generar mucha industria: la industria del asfalto, la industria del cemento, la industria del vidrio... Todo eso mueve el trabajo y convoca al trabajo de mucha gente."
   → -1: Public works spending justified as employment and industrial stimulus; expansionary fiscal stance.

8. "Extensión de Asignaciones Familiares, ampliación de la Asignación Universal por Hijo, la Pensión Universal al Adulto Mayor, la Tarifa Social Federal, la Cobertura Universal de Salud, la construcción de 3 mil jardines de infantes... la Ley de Reparación Histórica con la cual le estamos cumpliendo a nuestros jubilados."
   → -1: Expansion of social transfers, universal child allowance, pension increases, broad social spending.

9. "Construimos ya 80 mil casas, entregamos ya 80 mil casas, hay 150 mil casas en construcción también, y construimos 5.300 obras públicas... dos obras públicas y media por día concluimos en nuestra gestión."
   → -1: Celebrates large-scale public housing and infrastructure programme; expansionary public investment."""

SYSTEM_PROMPT = _BASE_RUBRIC + _FEW_SHOT_EXAMPLES

USER_PROMPT_TEMPLATE = """Score each of the following {n} paragraphs from a political speech.

Respond with ONLY a JSON array of {n} objects, one per paragraph IN ORDER:
[{{"id": <integer>, "score": <-1|0|1>, "reason": "<8–12 words summarising the fiscal direction>"}}]

Paragraphs:
{paragraphs}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_paragraph(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if last_stop > max_chars // 2:
        return cut[:last_stop + 1].strip() + " [...]"
    return cut.strip() + " [...]"


def estimate_cost(n_paragraphs: int, batch_size: int, model: str,
                  system_prompt_chars: int = 0) -> dict:
    n_batches        = max(1, n_paragraphs // batch_size)
    tokens_sys       = max(350, system_prompt_chars // 4)   # ~1 token per 4 chars
    tokens_para_in   = (160 + 30) * batch_size
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


def parse_score_response(raw: str, expected_ids: list) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
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
            pid = int(pid); score = int(score)
        except (TypeError, ValueError):
            continue
        if score not in (-1, 0, 1):
            score = 0
        result[pid] = {
            "llm_score":  score,
            "llm_reason": str(item.get("reason", ""))[:120],
        }
    return result


def score_batch(client, batch: list, model: str, retry_limit: int = 3) -> dict:
    para_lines = "\n".join(
        f"{i+1}. [id={row['para_id']}] {truncate_paragraph(str(row['text_para']))}"
        for i, row in enumerate(batch)
    )
    user_msg = USER_PROMPT_TEMPLATE.format(n=len(batch), paragraphs=para_lines)

    for attempt in range(retry_limit):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw    = resp.content[0].text
            parsed = parse_score_response(raw, [r["para_id"] for r in batch])
            missing = {r["para_id"] for r in batch} - set(parsed.keys())
            if missing:
                if attempt < retry_limit - 1:
                    time.sleep(2 ** attempt)
                    continue
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
                return {r["para_id"]: {"llm_score": 0, "llm_reason": "api_error"} for r in batch}

    return {r["para_id"]: {"llm_score": 0, "llm_reason": "retry_exhausted"} for r in batch}


def aggregate_monthly(para_df: pd.DataFrame) -> pd.DataFrame:
    """EPU-style aggregation — identical to llm_scoring.py."""
    fiscal = para_df[para_df["is_fiscal"] == True].copy()
    records = []
    for (ym, pres), grp in fiscal.groupby(["year_month", "president"], observed=True):
        P_t = len(grp)
        H_t = (grp["llm_score"] == 1).sum()
        D_t = (grp["llm_score"] == -1).sum()
        records.append({
            "year_month":     ym,
            "president":      pres,
            "n_speeches":     grp["speech_id"].nunique(),
            "n_fiscal_paras": int(P_t),
            "H_t_llm":        int(H_t),
            "D_t_llm":        int(D_t),
            "net_hawkish_llm": (H_t - D_t) / P_t if P_t > 0 else np.nan,
        })
    monthly = pd.DataFrame(records)
    monthly["ym_dt"] = pd.to_datetime(monthly["year_month"])
    monthly.sort_values(["ym_dt", "president"], inplace=True, ignore_index=True)
    lo = monthly["net_hawkish_llm"].quantile(0.025)
    hi = monthly["net_hawkish_llm"].quantile(0.975)
    monthly["net_hawkish_llm_wins"] = monthly["net_hawkish_llm"].clip(lo, hi)
    mu  = monthly["net_hawkish_llm_wins"].mean()
    sig = monthly["net_hawkish_llm_wins"].std()
    monthly["net_hawkish_llm_z"] = (monthly["net_hawkish_llm_wins"] - mu) / sig if sig > 0 else 0.0
    mu_r  = monthly["net_hawkish_llm"].mean()
    sig_r = monthly["net_hawkish_llm"].std()
    monthly["net_hawkish_llm_z_raw"] = (monthly["net_hawkish_llm"] - mu_r) / sig_r if sig_r > 0 else 0.0
    monthly.drop(columns=["ym_dt"], inplace=True)
    return monthly[monthly["president"].isin(PRES_ORDER)].copy()


def plot_validation(monthly_fs: pd.DataFrame, monthly_dict: pd.DataFrame) -> tuple:
    """Four-panel validation chart (few-shot LLM vs dictionary)."""
    merged = monthly_fs.merge(
        monthly_dict[["year_month", "president", "net_hawkish_z"]],
        on=["year_month", "president"], how="inner",
    )
    merged["ym_dt"] = pd.to_datetime(merged["year_month"])
    merged.sort_values("ym_dt", inplace=True, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    inaug = ["2015-12-10", "2019-12-10", "2023-12-10"]

    ax = axes[0, 0]
    for pres in PRES_ORDER:
        sub = merged[merged["president"] == pres].sort_values("ym_dt")
        ax.plot(sub["ym_dt"], sub["net_hawkish_llm_z"],
                color=PRES_COLORS[pres], linewidth=2.0, label=f"{pres} LLM (few-shot)")
        ax.plot(sub["ym_dt"], sub["net_hawkish_z"],
                color=PRES_COLORS[pres], linewidth=1.2, linestyle="--",
                alpha=0.6, label=f"{pres} Dict")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    for d in inaug:
        ax.axvline(pd.Timestamp(d), color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Few-shot LLM signal (solid) vs Dictionary signal (dashed)")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    all_valid  = merged.dropna(subset=["net_hawkish_llm_z", "net_hawkish_z"])
    r_pearson  = all_valid["net_hawkish_llm_z"].corr(all_valid["net_hawkish_z"])
    r_spearman = all_valid["net_hawkish_llm_z"].corr(all_valid["net_hawkish_z"], method="spearman")
    for pres in PRES_ORDER:
        sub = all_valid[all_valid["president"] == pres]
        ax.scatter(sub["net_hawkish_z"], sub["net_hawkish_llm_z"],
                   color=PRES_COLORS[pres], alpha=0.55, s=22, label=pres)
    lo = min(all_valid[["net_hawkish_z", "net_hawkish_llm_z"]].min())
    hi = max(all_valid[["net_hawkish_z", "net_hawkish_llm_z"]].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Dictionary z-score"); ax.set_ylabel("Few-shot LLM z-score")
    ax.set_title(f"Monthly scatter  r={r_pearson:.3f} (Pearson)  ρ={r_spearman:.3f} (Spearman)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PRES_COLORS[p], label=p) for p in PRES_ORDER], fontsize=9)

    ax = axes[1, 0]
    x = np.arange(len(PRES_ORDER)); width = 0.35
    means_fs   = [merged[merged["president"] == p]["net_hawkish_llm_z"].mean() for p in PRES_ORDER]
    means_dict = [merged[merged["president"] == p]["net_hawkish_z"].mean()     for p in PRES_ORDER]
    bars1 = ax.bar(x - width/2, means_fs,   width, label="LLM (few-shot)", color="#9C27B0", alpha=0.8)
    bars2 = ax.bar(x + width/2, means_dict, width, label="Dictionary",     color="#607D8B", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(PRES_ORDER)
    ax.set_title("Mean z-score by president: Few-shot LLM vs Dictionary")
    ax.set_ylabel("Mean z-score"); ax.legend(fontsize=9)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    full = merged.sort_values("ym_dt").dropna(
        subset=["net_hawkish_llm_z", "net_hawkish_z"]).reset_index(drop=True)
    roll_vals, roll_idx = [], []
    for i in range(5, len(full)):
        w = full.iloc[i-5:i+1]
        if w[["net_hawkish_llm_z", "net_hawkish_z"]].std().min() == 0:
            continue
        roll_vals.append(w["net_hawkish_llm_z"].corr(w["net_hawkish_z"]))
        roll_idx.append(full.iloc[i]["ym_dt"])
    roll_corr = pd.Series(roll_vals, index=roll_idx).dropna()
    ax.plot(roll_corr.index, roll_corr.values, color="#E91E63", linewidth=1.8)
    ax.axhline(0.65, color="green",  linewidth=1, linestyle="--", alpha=0.7, label="Target ρ=0.65")
    ax.axhline(0,    color="black",  linewidth=0.7, linestyle="--", alpha=0.4)
    ax.fill_between(roll_corr.index, roll_corr.values, 0,
                    where=(roll_corr.values >= 0), alpha=0.15, color="#E91E63")
    for d in inaug:
        ax.axvline(pd.Timestamp(d), color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Rolling 6-month Pearson correlation: Few-shot LLM vs Dictionary")
    ax.set_ylabel("Pearson r"); ax.set_ylim(-1.1, 1.1); ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "llm_vs_dict_signal_fewshot.png")
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

    print(f"Mode: FEW-SHOT (9 labeled calibration examples in system prompt)")
    print(f"  System prompt: {len(SYSTEM_PROMPT):,} chars  "
          f"(base: {len(_BASE_RUBRIC):,}  +  examples: {len(_FEW_SHOT_EXAMPLES):,})")
    print(f"  Example para IDs excluded from scoring: {sorted(FEW_SHOT_PARA_IDS)}")

    # ── 1. Load fiscal paragraphs ─────────────────────────────────────────────
    print(f"\nLoading {PARA_SCORED}...")
    if not os.path.exists(PARA_SCORED):
        raise FileNotFoundError(
            "paragraphs_scored.csv not found. Run signal/scoring/tfidf_dictionary.py first."
        )
    para_df = pd.read_csv(PARA_SCORED)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)].copy()
    fiscal  = para_df[para_df["is_fiscal"] == True].copy()

    # Exclude the 9 example paragraphs to avoid circularity
    n_before = len(fiscal)
    fiscal = fiscal[~fiscal["para_id"].isin(FEW_SHOT_PARA_IDS)].copy()
    n_excluded = n_before - len(fiscal)
    print(f"  Total paragraphs : {len(para_df):,}")
    print(f"  Fiscal paragraphs: {n_before:,}")
    print(f"  Excluded (examples): {n_excluded}  →  scoring: {len(fiscal):,}")
    print(f"  By president     : {dict(fiscal['president'].value_counts())}")

    if sample:
        fiscal = fiscal.sample(n=min(sample, len(fiscal)), random_state=42)
        print(f"  Sampled          : {len(fiscal):,} paragraphs")

    # ── 2. Cost estimate ──────────────────────────────────────────────────────
    est = estimate_cost(len(fiscal), batch_size, model, system_prompt_chars=len(SYSTEM_PROMPT))
    print(f"\n── Cost estimate ({model}) ─────────────────────────────────────")
    print(f"  Paragraphs   : {est['n_paragraphs']:,}")
    print(f"  API calls    : ~{est['n_batches']:,}  (batch size {batch_size})")
    print(f"  Input tokens : ~{est['tokens_in']:,}")
    print(f"  Output tokens: ~{est['tokens_out']:,}")
    print(f"  Estimated cost: ~${est['cost_usd']:.2f} USD")
    print(f"────────────────────────────────────────────────────────────────")

    if dry_run:
        print("\nDry run complete. Remove --dry-run to execute.")
        return

    # ── 3. Load checkpoint ────────────────────────────────────────────────────
    checkpoint: dict = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            checkpoint = json.load(f)
        already_done = sum(1 for k in checkpoint if checkpoint[k].get("llm_score") is not None)
        print(f"\nResuming from checkpoint: {already_done:,} paragraphs already scored")

    # ── 4. API client ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # ── 5. Score in batches ───────────────────────────────────────────────────
    rows       = fiscal.to_dict("records")
    to_score   = [r for r in rows if str(r["para_id"]) not in checkpoint]
    n_done     = len(rows) - len(to_score)

    print(f"\nScoring {len(to_score):,} paragraphs (skipping {n_done:,} already done)...")
    print(f"Model: {model}  |  Batch size: {batch_size}\n")

    batches = [to_score[i:i+batch_size] for i in range(0, len(to_score), batch_size)]
    start   = time.time()

    for i, batch in enumerate(batches):
        scores = score_batch(client, batch, model)
        for row in batch:
            pid = row["para_id"]
            checkpoint[str(pid)] = scores.get(pid, {"llm_score": 0, "llm_reason": "no_response"})
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)
        done_so_far = n_done + (i + 1) * batch_size
        pct     = min(100, done_so_far / len(rows) * 100)
        elapsed = time.time() - start
        eta     = (elapsed / max(i+1, 1)) * (len(batches) - i - 1)
        print(
            f"  Batch {i+1:4d}/{len(batches):4d}  [{pct:5.1f}%]  "
            f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m",
            end="\r", flush=True,
        )

    print(f"\nScoring complete. {len(checkpoint):,} paragraphs scored.")

    # ── 6. Merge scores ───────────────────────────────────────────────────────
    fiscal["llm_score"]  = fiscal["para_id"].apply(
        lambda pid: checkpoint.get(str(pid), {}).get("llm_score", 0)
    )
    fiscal["llm_reason"] = fiscal["para_id"].apply(
        lambda pid: checkpoint.get(str(pid), {}).get("llm_reason", "")
    )

    # ── 7. Monthly aggregation ────────────────────────────────────────────────
    print("\nAggregating to monthly (EPU-style)...")
    monthly_fs = aggregate_monthly(fiscal)
    print(f"  {len(monthly_fs)} month-president rows")

    print("\n── Few-shot signal means by president ───────────────────────────")
    for pres in PRES_ORDER:
        sub = monthly_fs[monthly_fs["president"] == pres]
        z   = sub["net_hawkish_llm_z"].dropna()
        print(f"  {pres:<6}  n={len(z):3d}  z={z.mean():+.3f}  std={z.std():.3f}")

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    para_out = os.path.join(INTERIM_DIR, "paragraphs_llm_scored_fewshot.csv")
    fiscal.to_csv(para_out, index=False)
    print(f"\nSaved: {para_out}")

    monthly_out = os.path.join(INTERIM_DIR, "monthly_signal_llm_fewshot.csv")
    monthly_fs.to_csv(monthly_out, index=False)
    print(f"Saved: {monthly_out}")

    if os.path.exists(MONTHLY_DICT):
        dict_sig = pd.read_csv(MONTHLY_DICT)[
            ["year_month", "president", "net_hawkish_z", "net_hawkish_z_raw",
             "net_hawkish_rob_z", "H_t", "D_t", "n_fiscal_paras", "n_speeches"]
        ]
        bvar = monthly_fs.merge(dict_sig, on=["year_month", "president"], how="left",
                                suffixes=("_llm", "_dict"))
        bvar_cols = [
            "year_month", "president",
            "net_hawkish_llm_z", "net_hawkish_z",
            "net_hawkish_llm_z_raw", "net_hawkish_z_raw",
            "H_t_llm", "D_t_llm", "H_t", "D_t",
            "n_fiscal_paras", "n_speeches",
        ]
        bvar_out = os.path.join(PROCESSED_DIR, "bvar_signal_llm_fewshot.csv")
        bvar[[c for c in bvar_cols if c in bvar.columns]].to_csv(bvar_out, index=False)
        print(f"Saved: {bvar_out}")

    # ── 9. Cross-validation against dictionary ────────────────────────────────
    r_pearson, r_spearman = None, None
    if os.path.exists(MONTHLY_DICT):
        print("\nCross-validating few-shot LLM vs dictionary signal...")
        monthly_dict = pd.read_csv(MONTHLY_DICT)
        r_pearson, r_spearman = plot_validation(monthly_fs, monthly_dict)
        print(f"  Pearson  r  = {r_pearson:.3f}")
        print(f"  Spearman ρ  = {r_spearman:.3f}")
        if r_spearman >= 0.65:
            print(f"  ✓ Target ρ ≥ 0.65 MET (few-shot vs dictionary)")
        else:
            print(f"  ✗ Target ρ ≥ 0.65 NOT met")

    # ── 10. Few-shot vs zero-shot stability comparison ────────────────────────
    r_fs_zs = r_rho_fs_zs = None
    if os.path.exists(MONTHLY_ZS):
        print("\nComparing few-shot vs zero-shot monthly series...")
        monthly_zs = pd.read_csv(MONTHLY_ZS)
        cmp = monthly_fs.merge(
            monthly_zs[["year_month", "president", "net_hawkish_llm_z"]],
            on=["year_month", "president"], how="inner",
            suffixes=("_fs", "_zs"),
        ).dropna(subset=["net_hawkish_llm_z_fs", "net_hawkish_llm_z_zs"])

        if len(cmp) > 0:
            r_fs_zs     = cmp["net_hawkish_llm_z_fs"].corr(cmp["net_hawkish_llm_z_zs"])
            r_rho_fs_zs = cmp["net_hawkish_llm_z_fs"].corr(
                cmp["net_hawkish_llm_z_zs"], method="spearman"
            )
            print(f"  n months compared : {len(cmp)}")
            print(f"  Pearson  r  (few-shot vs zero-shot) = {r_fs_zs:.3f}")
            print(f"  Spearman ρ  (few-shot vs zero-shot) = {r_rho_fs_zs:.3f}")
            if r_rho_fs_zs > 0.85:
                print(f"  ✓ Stability target ρ > 0.85 MET — zero-shot results are robust")
            else:
                print(f"  ✗ Stability target ρ > 0.85 NOT met — investigate prompt sensitivity")
        else:
            print("  [no overlapping months between few-shot and zero-shot series]")
    else:
        print(f"\n  [zero-shot file not found — run llm_scoring.py first: {MONTHLY_ZS}]")

    # ── 11. Audit report ──────────────────────────────────────────────────────
    lines = [
        "=== LLM SCORING SUMMARY (Stage 5b — FEW-SHOT ROBUSTNESS) ===",
        f"Model          : {model}",
        f"Mode           : Few-shot (9 labeled examples in system prompt)",
        f"Scored         : {len(fiscal):,} fiscal paragraphs",
        f"Excluded (examples): {n_excluded} para IDs — {sorted(FEW_SHOT_PARA_IDS)}",
        f"Batches        : {len(batches):,}  (batch size {batch_size})",
        "",
        "── FEW-SHOT LLM SCORE DISTRIBUTION BY PRESIDENT ─────────────────────",
        f"  {'President':<8} {'N':>5}  {'Hawkish':>8}  {'Neutral':>8}  {'Dovish':>8}  "
        f"{'H%':>6}  {'D%':>6}",
    ]
    for pres in PRES_ORDER:
        sub = fiscal[fiscal["president"] == pres]
        n   = len(sub)
        if n == 0:
            continue
        h   = (sub["llm_score"] == 1).sum()
        neu = (sub["llm_score"] == 0).sum()
        d   = (sub["llm_score"] == -1).sum()
        lines.append(
            f"  {pres:<8} {n:>5}  {h:>8}  {neu:>8}  {d:>8}  "
            f"{h/n*100:>5.1f}%  {d/n*100:>5.1f}%"
        )

    lines += [
        "",
        "── MONTHLY SIGNAL SUMMARY (FEW-SHOT LLM) ────────────────────────────",
        f"  {'President':<8} {'N':>5}  {'Mean-z':>8}  {'Std-z':>8}  {'Min-z':>8}  {'Max-z':>8}",
    ]
    for pres in PRES_ORDER:
        sub = monthly_fs[monthly_fs["president"] == pres]
        z   = sub["net_hawkish_llm_z"].dropna()
        lines.append(
            f"  {pres:<8} {len(z):>5}  {z.mean():>+8.3f}  {z.std():>8.3f}  "
            f"{z.min():>+8.3f}  {z.max():>+8.3f}"
        )

    if r_pearson is not None:
        lines += [
            "",
            "── CROSS-VALIDATION: FEW-SHOT LLM vs DICTIONARY ─────────────────────",
            f"  Pearson  r  = {r_pearson:.4f}",
            f"  Spearman ρ  = {r_spearman:.4f}",
            f"  Target ρ ≥ 0.65: {'MET' if r_spearman >= 0.65 else 'NOT MET'}",
        ]

    if r_fs_zs is not None:
        lines += [
            "",
            "── KEY ROBUSTNESS CHECK: FEW-SHOT vs ZERO-SHOT ──────────────────────",
            f"  Pearson  r  = {r_fs_zs:.4f}",
            f"  Spearman ρ  = {r_rho_fs_zs:.4f}",
            f"  Stability target ρ > 0.85: {'MET' if r_rho_fs_zs > 0.85 else 'NOT MET'}",
            "",
            f"  {'President':<8} {'Mean-z (FS)':>12}  {'Mean-z (ZS)':>12}  {'Diff':>8}",
        ]
        for pres in PRES_ORDER:
            sub_cmp = cmp[cmp["president"] == pres]
            if len(sub_cmp) == 0:
                continue
            mfs = sub_cmp["net_hawkish_llm_z_fs"].mean()
            mzs = sub_cmp["net_hawkish_llm_z_zs"].mean()
            lines.append(f"  {pres:<8} {mfs:>+12.3f}  {mzs:>+12.3f}  {mfs-mzs:>+8.3f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    path = os.path.join(TABLES_DIR, "llm_scoring_summary_fewshot.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved: {path}")

    return fiscal, monthly_fs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5b: Few-shot robustness check for LLM fiscal hawkishness signal."
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
