# Masters-Project — Claude Handoff Document
**Last updated:** 2026-05-14 (econometric approach updated to DFM → residual shock → LP)  
**Author:** Rayyaan Kazi (BSE Master's student)  
**Project:** Fiscal hawkishness NLP signal from Argentine presidential speeches. Primary identification: Dynamic Factor Model of macro variables → regress signal on factors → use residuals as purged communication shock in Local Projections (Jordà 2005). BVAR retained as robustness check.

---

## Project Goal

Construct a monthly fiscal hawkishness signal from Argentine presidential speeches (Macri 2015–2019, Alberto Fernández 2019–2023, Milei 2023–2026). The primary identification strategy is a two-stage procedure: (1) estimate a Dynamic Factor Model (DFM) of macro variables to summarise economic conditions; (2) regress the hawkishness signal on the DFM factors and use the residuals as a purged communication shock series in Local Projections (Jordà 2005). Regime heterogeneity is tested via a Milei interaction term. BVAR is retained as a robustness check.

**Key identification rationale:** The raw signal captures both the president's endogenous response to economic conditions and genuine communication surprises. Projecting the signal onto DFM factors strips out the predictable component, leaving the unexpected hawkishness shock. This follows Bernoth (2025), who constructs communication shocks as residuals from a regression of the stance indicator on macro-financial variables.

**Primary LP input:** `net_hawkish_llm_z` in `data/interim/monthly_signal_llm.csv`  
**Robustness/replication check:** `net_hawkish_z` in `data/interim/monthly_signal.csv`

---

## Pipeline Overview

```
Stage 1: Scraping       signal/scraping/scraper.py                    ✓ COMPLETE
           ↓
Stage 2: Raw corpus     data/raw/speeches_raw.csv                     ✓ COMPLETE
           ↓
Stage 3: LDA            signal/topic_modeling/lda.py                  ✓ COMPLETE
           ↓  paragraphs_lda.csv (validation/narrative only — NOT load-bearing filter)
Stage 4: Dictionary     signal/scoring/tfidf_dictionary.py            ✓ COMPLETE (v8)
           ↓  paragraphs_scored.csv, speeches_scored.csv, monthly_signal.csv
Stage 4b: Keyword audit  signal/validation/                           ← PENDING (label ~150 paras)
           ↓  filter precision/recall report
Stage 5: LLM scoring    signal/scoring/llm_scoring.py                 ✓ COMPLETE (v8, $1.09)
           ↓  paragraphs_llm_scored.csv, monthly_signal_llm.csv, bvar_signal_llm.csv
Stage 5b: Few-shot      signal/scoring/llm_scoring.py --few-shot      ← NEXT
           ↓  few-shot robustness run (compare ρ with zero-shot)
Stage 5c: Validation    signal/validation/                            ✓ COMPLETE (2026-05-13)
           ↓  72 human-labeled paragraphs; macro F1=0.831, κ=0.750, accuracy=0.833
Stage 6: External valid (not yet built)                               ← PENDING
           ↓  correlate net_hawkish_llm_z with primary balance, EMBI+, ARS/USD
Stage 7a: DFM              econometrics/identification/               ← IN PROGRESS (group)
           ↓  Dynamic Factor Model of macro variables (inflation, EMAE, fiscal balance)
           ↓  Extract 1–2 factors summarising economic conditions
Stage 7b: Shock extraction econometrics/identification/               ← IN PROGRESS (group)
           ↓  Regress net_hawkish_llm_z on DFM factors → residuals = purged shock
           ↓  NOTE: generated regressor — must bootstrap full two-stage procedure
Stage 7c: Local Projections econometrics/estimation/                  ← IN PROGRESS (group)
           ↓  Jordà (2005) LP-IRFs using residual shock series
           ↓  LP spec: y_{t+h} − y_{t−1} (cumulative changes) — NOT ∆y_{t+h}
           ↓  Milei interaction term: shock_t × Milei_t
           ↓  HAC SEs (lags=4) + wild bootstrap (B=500, Rademacher) — bootstrap covers full two-stage
           ↓  BVAR retained as robustness
```

---

## Key Files

| File | Description |
|------|-------------|
| `data/raw/speeches_raw.csv` | Raw speech corpus |
| `data/interim/paragraphs_lda.csv` | Paragraph-level with LDA topic probabilities |
| `data/interim/paragraphs_scored.csv` | Paragraph-level with dictionary hit counts + `is_fiscal` flag |
| `data/interim/speeches_scored.csv` | Speech-level aggregated dictionary scores |
| `data/interim/monthly_signal.csv` | Monthly dictionary signal (robustness check) |
| `data/interim/paragraphs_llm_scored.csv` | Fiscal paragraphs with LLM scores (-1/0/+1) |
| `data/interim/monthly_signal_llm.csv` | **Monthly LLM signal — primary BVAR input** |
| `data/interim/llm_scores_checkpoint.json` | LLM scoring checkpoint (resume support) |
| `data/processed/bvar_signal.csv` | Clean dictionary signal, BVAR-ready |
| `data/processed/bvar_signal_llm.csv` | Clean LLM + dictionary signal, BVAR-ready |
| `signal/dictionaries/hawkish_terms.txt` | 61 hawkish terms (v6) — used in Stage 4 only |
| `signal/dictionaries/dovish_terms.txt` | 46 dovish terms (v6) — used in Stage 4 only |
| `signal/scoring/tfidf_dictionary.py` | Stage 4 scoring pipeline (v8 — keyword fiscal filter) |
| `signal/scoring/llm_scoring.py` | Stage 5 LLM paragraph scoring (Claude API) |
| `signal/topic_modeling/wordclouds.py` | Word cloud generator — reads `paragraphs_scored.csv`, filters by `is_fiscal` (v8 BBD flag) |
| `outputs/figures/wc_fiscal_<president>.png` | Fiscal-filtered word clouds (v8) — used in presentation slides |
| `outputs/figures/llm_vs_dict_signal.png` | 4-panel LLM vs dictionary validation chart — appendix slide |
| `outputs/tables/scoring_summary.txt` | Dictionary scoring audit |
| `outputs/tables/llm_scoring_summary.txt` | LLM scoring audit + cross-validation stats |
| `signal/validation/paragraphs_to_label.xlsx` | Completed human labeling file (72 paragraphs) |
| `signal/validation/labeling_key.csv` | Private key: LLM scores + president for each labeled paragraph |
| `signal/validation/human_labels.csv` | Merged human + LLM labels — archival record |
| `signal/validation/sample_for_labeling.py` | Sampling script — stratified draw of 72 paragraphs |
| `signal/validation/evaluate_labels.py` | Evaluation script — computes confusion matrix, F1, κ |
| `signal/validation/plot_validation.py` | Validation figure generator (4 charts) |
| `outputs/figures/validation_confusion_matrix.png` | Human vs LLM confusion matrix — main validation figure |
| `outputs/figures/validation_class_metrics.png` | Per-class precision/recall/F1 bar chart |
| `outputs/figures/validation_president_breakdown.png` | Accuracy/F1/κ by president |
| `outputs/figures/validation_error_analysis.png` | Correct/adjacent/extreme error breakdown |
| `outputs/tables/human_validation_report.txt` | Full validation report (text) |

---

## CRITICAL: What the LLM Scoring Does and Does NOT Use

The LLM scoring pipeline (Stage 5) is **completely independent** of the hawkish/dovish dictionaries.

**Stage 5 uses:**
- Raw paragraph text (sent directly to Claude API)
- Fiscal keyword filter (22-word FISCAL_KEYWORDS list, v8) to decide which paragraphs are scored
- A rubric prompt (hawkish +1 / neutral 0 / dovish -1)

**Stage 5 does NOT use:**
- The hawkish_terms.txt dictionary (61 terms)
- The dovish_terms.txt dictionary (46 terms)
- LDA topic probabilities
- Any other preprocessing

The dictionaries are exclusively Stage 4 (tfidf_dictionary.py). The only shared component between Stage 4 and Stage 5 is the fiscal keyword filter (v8, 22 keywords).

This independence is the key validation property: the LLM and dictionary signals agree at ρ = 0.835 despite using completely different methods.

---

## Signal Results — Current State

### LLM Signal (Primary — Stage 5, zero-shot)

**Model:** claude-haiku-4-5-20251001  
**Paragraphs scored:** 3,904 fiscal keyword-filtered paragraphs (v8 filter)  
**Aggregation:** EPU-style (H_t − D_t) / P_t, winsorised 2.5/97.5 pct, cross-president z-scored

| President | N months | Mean-z | Std-z | Score distribution (H/N/D) |
|-----------|----------|--------|-------|----------------------------|
| Macri | 49 | +0.179 | 0.622 | 28.8% / 51.6% / 19.6% |
| AF | 47 | −0.983 | 0.382 | 7.1% / 41.5% / 51.4% |
| Milei | 29 | +1.291 | 0.346 | 72.9% / 26.7% / 0.4% |

Milei–AF gap: **2.27 z-units**  
January 2024 (Davos): **confirmed** — LLM scores z = +1.542 (correctly hawkish; dictionary z = +0.238)

**Note on Macri v7→v8 shift (+0.38z → +0.18z):** The drop reflects two effects: (1) `pobreza` added ~339 neutral paragraphs to Macri's P_t (confirmed by spot-check — all scored 0/0), diluting the denominator; (2) `jubil`, `pension`, `salario` added ~109 genuinely dovish paragraphs (Macri's social contract commitments on pensions/wages that v7 missed). The direction ordering is unchanged and the v8 reading is more accurate.

### Dictionary Signal (Robustness — Stage 4, v8)

| President | N months | Mean-z | Std-z |
|-----------|----------|--------|-------|
| Macri | 49 | +0.212 | 0.460 |
| AF | 47 | −0.979 | 0.579 |
| Milei | 29 | +1.228 | 0.514 |

### Cross-Validation (v8, n=123 common months)

| Comparison | Pearson r | Spearman ρ |
|------------|-----------|------------|
| LLM v8 vs Dictionary v8 | 0.839 | 0.835 |
| LLM v8 vs Old signal | 0.740 | 0.670 |
| Dictionary v8 vs Old signal | 0.713 | 0.686 |

Target ρ ≥ 0.65: **MET on all comparisons**. Primary cross-validation (LLM vs Dictionary) improved substantially from v7 (ρ = 0.765 → 0.835), reflecting the more symmetric v8 filter.

**Note on Old signal correlations:** The old signal uses within-president z-scoring (Macri mean = −0.82, AF mean = −1.14, Milei mean = +1.03), making level comparisons invalid. The ρ = 0.67–0.69 reflects agreement in month-to-month movements, not levels.

**Key divergences (v8):** January 2024 Milei gap = 1.30z (LLM +1.542 vs Dict +0.238) — LLM correctly captures Davos ideological hawkishness. Largest overall divergence: Macri 2016-05 (LLM −1.228 vs Dict +0.325, gap = 1.55z) — likely reflects gradualismo framing where fiscal vocabulary was present but stance was non-committal.

---

## Fiscal Filter — Keyword Filter (v8)

**Method:** Baker, Bloom & Davis (2016) EPU keyword-in-text approach, directly following their Appendix B fiscal category methodology. A paragraph is classified as fiscal if it contains at least one of 22 keywords (stems — each matches all morphological variants via `\b<stem>\w*`).

**v8 keywords (22 total):**
`deficit, superavit, gasto, presupuest, fiscal, impuesto, deuda, inflacion, ajuste, subsidio, recaudacion, austeridad, obra publica, inversion publica, inversion social, emision, jubil, pension, salario, fmi, privatiz, pobreza`

**v7 → v8 additions (7 new terms):**
| Keyword | Captures | Rationale |
|---------|----------|-----------|
| `emision` | emisión monetaria, emitir | Core Milei anti-inflation argument; missed in Davos/CPAC speeches |
| `jubil` | jubilados, jubilación | Largest spending line; classic dovish content for AF |
| `pension` | pensión, pensiones | Overlaps with jubil; catches distinct phrasings |
| `salario` | salario, salarial | Real wage protection — AF's dominant dovish frame |
| `fmi` | FMI, Fondo Monetario | IMF programme targets central to all three administrations |
| `privatiz` | privatización, privatizar | Milei state reform; hawkish not otherwise caught |
| `pobreza` | pobreza, pobres | Social spending justification; dovish content gap |

This replaced the LDA threshold filter (v6) which had an asymmetric bias — it captured ~84% of hawkish hits but only ~40% of dovish hits, because hawkish terms are inherently fiscal-accounting vocabulary while dovish terms include social-spending vocabulary assigned by LDA to a different topic. BBD's symmetric keyword approach fixes this.

**v8 rerun complete (2026-05-02).** Both tfidf_dictionary.py and llm_scoring.py have been re-run. All signal files are now v8.

**Fiscal paragraph counts by president (v8):**
- Macri: 873 fiscal paragraphs / 5,247 total (16.6%) — 17.8 fiscal paras/month
- AF: 1,140 fiscal paragraphs / 6,264 total (18.2%) — 24.3 fiscal paras/month
- Milei: 1,891 fiscal paragraphs / 4,686 total (40.4%) — 65.2 fiscal paras/month

**v7 → v8 change:** Macri +93% (driven by `pobreza` +339 neutral paras, `jubil` +53), AF +38%, Milei +15%. Macri increase confirmed as mostly neutral dilution via spot-check (15/15 sampled `pobreza`-only paras scored 0/0 by dictionary).

**Milei speech note:** Milei gives fewer speeches than Macri/AF (6.4/month vs 13.4/month) but his speeches are longer and have far higher fiscal content density (35% vs 8–13%). The LLM signal level difference reflects genuine fiscal discourse intensity, not speech volume.

**BBD audit requirement:** BBD validate their keyword filter by human-reading 12,000 newspaper articles and reporting precision/recall. The equivalent here is to label ~100–150 paragraphs as fiscal/non-fiscal and compute filter precision/recall. This audit doubles as the human validation holdout for LLM scoring (label both fiscal/non-fiscal AND hawkish/neutral/dovish in one pass).

**Currently excluded: 1,593 ideology-only paragraphs** (is_ideology=True, is_fiscal=False):
- Milei: 1,106 (23.6% of his total paras)
- AF: 369 (5.9%)
- Macri: 118 (2.2%)

Samples show Macri/AF ideology-only paragraphs are mostly non-fiscal (diplomacy, human rights, democracy). Milei's include some genuine hawkish fiscal content (property rights arguments, monetary emission criticism using non-keyword vocabulary). The v8 `emision` keyword will now catch some of these.

**Embedding filter approach (considered, rejected):** An Ash & Hansen (2023) semantic similarity filter using multilingual-E5-base embeddings was implemented and tested. It failed due to the anisotropy problem in transformer embeddings — all paragraphs scored above the highest threshold tested (0.45), meaning the model could not discriminate fiscal from non-fiscal text at any threshold without fine-tuning on labeled pairs. The BBD keyword approach was retained as simpler, more transparent, and directly citable.

---

## LLM Scoring Methodology — Literature Grounding

### Why LLM-primary over dictionary-primary

The dictionary has a structural limitation: Milei's most important speeches (Davos Jan 2024, WEF, UN General Assembly) communicate fiscal stance through philosophical argument rather than fiscal-accounting vocabulary. The LLM evaluates fiscal *intent* from context.

**Speaker identity is never shown to the model** — no president name, speech title, or date in the prompt. The model scores content, not speaker. This avoids the motosierra/licuadora circularity problem entirely.

### Literature basis for the approach

| Paper | Approach | Key insight for our project |
|-------|----------|----------------------------|
| Hansen & Kazinnik (2024) | Prompted GPT for FOMC sentences | Validates API-prompting (not fine-tuning) for central bank text; few-shot examples improve accuracy |
| Bernoth (2025, DIW dp2137) | Fine-tuned RoBERTa, BVAR | Same EPU formula; communication SHOCK = residual from regressing signal on macro variables |
| IMF (2025, WP 2025/109) | Fine-tuned LLM, sentence-level | Scale and multilingual validation of LLM classification approach |
| Bank of England (2025, SWP 1127) | "Tens-of-shot" embeddings | Only need ~20–30 labeled examples for strong classification; validates few-shot approach |
| Baker, Bloom & Davis (2016) | EPU keyword counting | Foundation for keyword fiscal filter and (H−D)/P aggregation formula |
| Gentzkow, Kelly & Taddy (2019) | Text-as-data survey (JEL) | Document unit choice (paragraph vs sentence) should be checked for sensitivity |

### Zero-shot vs few-shot design

**Current implementation:** Zero-shot (no labeled examples in prompt)  
**Planned robustness:** Few-shot with 9 labeled examples (3 per class) drawn from actual corpus

Rationale: Zero-shot is the primary result — pure blind classification, no anchoring. Few-shot robustness shows results are stable when the model is given Argentine-corpus-specific examples including ideological hawkish content (Davos-style arguments).

**Proposed few-shot examples (candidates identified, not yet implemented):**

*Hawkish (+1) — fiscal accounting:*
- Milei 2024-09: "nuestro compromiso con el equilibrio fiscal es inquebrantable y no estamos dispuestos a negociar... el equilibrio fiscal"
- Milei 2024-08: "nuestro compromiso férreo con una política de emisión cero... dejemos de padecer la inflación"

*Hawkish (+1) — ideological framing (critical for capturing Davos-type content):*
- Para 4633: "el Estado se financia a través de impuestos y los impuestos se cobran de manera coactiva. A mayor carga impositiva mayor es la coacción"

*Dovish (-1):*
- AF 2021-09: obra pública generating employment
- Macri 2016-10: AUH expansion, social tariffs (cross-president dovish)
- AF 2023-01: public works housing programme

*Neutral (0):*
- AF 2022-01: IMF negotiation description without direction
- AF 2022-10: budget figures without endorsement
- Macri 2019-06: company export growth, structural description

### Human validation — COMPLETE (2026-05-13)

72 fiscal paragraphs labeled manually (stratified: 24 per class × 3 presidents), blinded — no president name, date, or LLM score shown during labeling. ~25% of sample drawn from LLM-dictionary disagreement cases (hardest cases).

**Results:**

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.833 |
| Macro F1 | 0.831 |
| Cohen's κ | 0.750 |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Dovish (−1) | 0.875 | 0.840 | 0.857 |
| Neutral (0) | 0.708 | 0.773 | 0.739 |
| Hawkish (+1) | 0.917 | 0.880 | 0.898 |

| President | N | Accuracy | Macro F1 | κ |
|-----------|---|----------|----------|---|
| Macri | 18 | 0.833 | 0.838 | 0.737 |
| AF | 26 | 0.846 | 0.866 | 0.690 |
| Milei | 28 | 0.821 | 0.743 | 0.616 |

**Key finding:** Zero extreme errors (no dovish↔hawkish misclassifications). All 12 errors are adjacent-class (neutral↔hawkish or neutral↔dovish). LLM-dict disagreement cases perform identically to agreement cases (both 0.833 accuracy).

**Interpretation:** κ = 0.750 is moderate-strong agreement. Neutral F1 = 0.739 is the weak point, consistent with zero-shot LLM behaviour on residual categories. Milei κ = 0.616 is the lowest by president but above the 0.60 minimum; no systematic directional bias detected.

**Figures:** `outputs/figures/validation_*.png` (4 charts — confusion matrix, per-class metrics, president breakdown, error analysis).

**Note on BBD filter audit:** The human validation sample was drawn exclusively from fiscal paragraphs (is_fiscal=True), so it validates LLM scoring accuracy but not keyword filter precision/recall. A separate filter audit (sampling from both fiscal and non-fiscal paragraphs) remains pending — see What Still Needs Doing.

---

## Scoring Pipelines — Technical Notes

### Stage 4: tfidf_dictionary.py (v8)

- **Keyword fiscal filter (v8):** 22 keywords — BBD (2016) Appendix B approach; replaces LDA threshold (v6) and v7 15-keyword list
- **EPU paragraph counting:** each fiscal para casts binary vote has_hawkish / has_dovish (Baker, Bloom & Davis 2016)
- **Winsorisation:** 2.5/97.5 pct before z-scoring
- **Two-tier negation:** Tier 1 strong (10-word), Tier 2 weak (3-word)
- **Direction-aware elimination:** suppresses dovish hits only
- **Negation suppression rate (v8):** Macri 2.4% H / 2.4% D; AF 0.0% H / 2.7% D; Milei 2.4% H / 10.4% D. Milei's high dovish negation (10.4%) reflects his frequent use of "eliminar obra pública / subsidios" framing — direction-aware negation correctly suppresses these.

### Stage 5: llm_scoring.py

```bash
cd ~/Desktop/Masters-Project
source .venv/bin/activate
# On zsh/fish, prefix the key inline rather than using export:
ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring.py --dry-run
ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring.py

# v8 run stats: 3,904 paragraphs, 391 batches, ~$1.09, ~28 min
# Higher accuracy model (optional):
ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring.py --model claude-sonnet-4-6
```

Checkpoint/resume: `data/interim/llm_scores_checkpoint.json` — delete to rescore from scratch. For a clean rescore (e.g. filter change), delete checkpoint before running.

---

## What Still Needs Doing (Priority Order)

### BLOCKING — required before thesis defense

1. **Fix scoring_summary.txt header** — still says "v7" in the output title line. One-line fix in the script's summary text before next rerun, or edit the file directly.

### Signal robustness (complete before econometrics)

2. **Few-shot robustness run** ← NEXT TASK — Add 9 labeled examples (3 per class) to prompt in `llm_scoring.py`. Delete checkpoint. Re-run (~$1.09, ~$3.27 for Sonnet). Compare monthly ρ against zero-shot baseline (target ρ > 0.85). Candidate examples already identified in the "Zero-shot vs few-shot design" section. Also run with `--model claude-sonnet-4-6` for model-version robustness (both runs in same session, ~$4.36 total using batch API discount).

3. **BBD keyword filter audit** — separate from LLM validation. Sample ~100 paragraphs from full corpus (mix of is_fiscal=True and is_fiscal=False), label each as genuinely fiscal/non-fiscal, compute filter precision/recall. Pending — was not combined with LLM validation (which was done on fiscal-only sample).

4. **Ideology paragraph spot-check** — 1,593 paragraphs still excluded (Milei: 1,106). Sample ~50 Milei ideology-only paragraphs post-v8 to verify `emision` keyword now catches the main false negatives. If >20% of remaining excluded Milei paragraphs score ±1, reconsider inclusion.

5. **Sentence-level robustness** — Split fiscal paragraphs into sentences, score with LLM, compare monthly aggregations. If ρ > 0.85 vs paragraph-level, document and move on (Gentzkow, Kelly & Taddy 2019).

### Data issues to document explicitly

6. **Two missing AF months** — January 2020 and November 2023 have zero fiscal paragraphs. **Decision (2026-05-14): scored as 0** — absence of fiscal communication treated as neutral stance, not imputed. Stated in thesis data section (Section 3.1). Rationale: imputation conflates absence with balance; zero is more conservative.

7. **February 2026 anomaly** — Milei has P_t=1 (single fiscal paragraph that month: "saldando una deuda histórica" — ceremonial speech about San Martín's sword, caught by `deuda` keyword). Signal = −0.113z, anomalously low. **Decision (2026-05-14): retained in signal series and estimation sample.** Stated in thesis data section (Section 3.1).

### External validation (Stage 6)

8. **Correlate signal with Argentine macro data:**
   - Primary balance/GDP (Ministry of Economy) — expect positive correlation with hawkishness
   - EMBI+ Argentina (JPMorgan) — expect negative correlation (hawkish → lower risk)
   - ARS/USD parallel rate — expect negative correlation (hawkish → peso appreciation)

### Econometrics (Stage 7) — being handled by group

9. **DFM → residual shock → LP** — Updated identification strategy (2026-05-14):
   - **Stage 1 — DFM:** Fit a Dynamic Factor Model to parsimonious set of macro variables: inflation (INDEC IPC), EMAE (monthly GDP proxy), fiscal balance. Do NOT include EMBI+ or ARS/USD in DFM — these are outcome variables affected by communication. Extract 1–2 factors.
   - **Stage 2 — Shock extraction:** Regress `net_hawkish_llm_z` on DFM factors. Residuals = purged fiscal communication shock. This strips out the endogenous component (president responding to economic conditions).
   - **Stage 3 — LP:** For each outcome variable y and horizon h:
     `y_{t+h} − y_{t−1} = αh + βh·ε̂_t + δh·(ε̂_t × Milei_t) + γh·controls_t + u_{t+h}`
     where ε̂_t are the DFM residuals. LHS is **cumulative change** — NOT `∆y_{t+h}`.
   - **CRITICAL:** ε̂_t are generated regressors. Wild bootstrap (B=500, Rademacher) must cover the full two-stage procedure, not just the LP stage.
   - δh directly estimates the Milei regime difference at each horizon
   - Use `llm_signal_v8.csv` (delivered to group 2026-05-02)
   - BVAR retained as robustness check (not primary result)
   - Literature basis: Bernoth (2025) Section 5 — communication shock as residual from macro regression

### Presentation

10. **Mid-project meeting** — 2026-05-06. LaTeX source in `paper/` folder.
    - Slide order: Corpus → **Word clouds (fiscal-filtered)** → Pipeline → Signal results → ...
    - Word cloud slide uses `wc_fiscal_*.png` (v8 BBD filter, updated 2026-05-05)
    - LDA moved to appendix (descriptive only; not load-bearing)
    - LP is primary results frame; BVAR shown as robustness
    - LP slide equation needs updating to `y_{t+h} − y_{t−1}` before final version

### Documentation

11. **Git commit:**
    ```bash
    cd ~/Desktop/Masters-Project
    git add signal/ CLAUDE.md README.md outputs/ paper/
    git commit -m "v8+pres: fiscal wordclouds updated, LP primary, LDA to appendix"
    ```

---

## Running the Pipeline

```bash
cd ~/Desktop/Masters-Project
source .venv/bin/activate

# Stage 4 — dictionary signal (re-run if dictionaries changed)
python signal/scoring/tfidf_dictionary.py

# Stage 5 — LLM signal (needs API key)
export ANTHROPIC_API_KEY="sk-ant-..."
python signal/scoring/llm_scoring.py --dry-run   # check cost first
python signal/scoring/llm_scoring.py             # full run

# Outputs:
#   data/interim/paragraphs_scored.csv       (Stage 4)
#   data/interim/monthly_signal.csv          (Stage 4 — dictionary)
#   data/processed/bvar_signal.csv           (Stage 4 — clean BVAR)
#   data/interim/paragraphs_llm_scored.csv   (Stage 5)
#   data/interim/monthly_signal_llm.csv      (Stage 5 — LLM)
#   data/processed/bvar_signal_llm.csv       (Stage 5 — combined BVAR)
#   outputs/figures/llm_vs_dict_signal.png   (Stage 5 — validation chart)
```

---

## Key Numbers to Know

- **Corpus:** 1,498 speeches, 16,197 paragraphs, 3 presidents
- **Dictionary:** 61 hawkish + 46 dovish = 107 terms (Stage 4 only — not used in LLM scoring)
- **Fiscal paragraphs scored by LLM:** 3,904 (873 Macri + 1,140 AF + 1,891 Milei) — v8 filter
- **LLM primary LP input column:** `net_hawkish_llm_z` in `data/interim/monthly_signal_llm.csv`
- **Dictionary robustness column:** `net_hawkish_z` in `data/interim/monthly_signal.csv`
- **Cross-validation:** Pearson r = 0.839, Spearman ρ = 0.835 (LLM v8 vs dictionary v8)
- **January 2024:** LLM z = +1.542, Dict z = +0.238 — Davos correctly hawkish in both; LLM stronger
- **President ordering (LLM v8):** Milei +1.29z > Macri +0.18z > AF −0.98z (gap = 2.27z)
- **Ideology-only paragraphs (excluded):** 1,593 — Milei 1,106 / AF 369 / Macri 118
- **Fiscal keyword filter:** 22 keywords v8 (symmetric — BBD 2016 approach; rerun complete 2026-05-02)
- **Known anomaly:** Feb 2026 Milei P_t=1 (ceremonial speech, `deuda` false positive) → signal = −0.113z; flag as thin month
- **Missing months:** AF Jan 2020 and Nov 2023 absent (zero fiscal paragraphs); treatment undocumented — must state in thesis
- **Dead dictionary term:** `actualizacion tarifaria` — zero hits across full corpus; remove or note in audit
- **Shared signal file:** `llm_signal_v8.csv` at project root — delivered to econometrics group 2026-05-02
- **LLM model used:** claude-haiku-4-5-20251001, temperature=0
- **Human validation (2026-05-13):** 72 paragraphs labeled; accuracy=0.833, macro F1=0.831, κ=0.750; zero extreme errors; all errors adjacent-class; by president: Macri κ=0.737 / AF κ=0.690 / Milei κ=0.616

---

## Literature Basis

- **Fiscal keyword filter:** Baker, Bloom & Davis (2016) EPU keyword-in-text approach, Appendix B fiscal category — direct methodological basis for the 22-keyword filter and human audit requirement
- **Filter design / text methods survey:** Ash & Hansen (2023) *Text Algorithms in Economics*, Annual Review of Economics — embedding approach considered but rejected (anisotropy); BBD keyword approach retained
- **Aggregation formula:** Baker, Bloom & Davis (2016) (H_t − D_t) / P_t EPU paragraph-counting formula
- **LLM scoring:** Hansen & Kazinnik (2024) prompted LLM for central bank text (FOMC); Bank of England (2025) SWP 1127 tens-of-shot classification; IMF (2025) WP 2025/109 large-scale LLM central bank speech scoring
- **DFM identification:** Bernoth (2025, DIW dp2137) Section 5 — communication shock as residual from regressing stance indicator on macro-financial variables; Stock & Watson (2002) dynamic factor models
- **BVAR framework:** Bernoth (2025, DIW dp2137) BVAR specification; Istrefi & Piloiu (2014) inflation expectations BVAR — retained as robustness
- **Communication shock / generated regressors:** bootstrap must cover full two-stage DFM → LP procedure
- **Dictionary basis:** Blanchard & Leigh (2013) fiscal multipliers; Dornbusch & Edwards (1991) populist cycles; Alesina & Ardagna (2010); Kopits & Symansky (1998) fiscal rules; Barro-Gordon credibility
- **Document unit sensitivity:** Gentzkow, Kelly & Taddy (2019) text-as-data survey — basis for sentence-level robustness check
- **TVP-VAR:** Primiceri (2005) time-varying structural VARs
