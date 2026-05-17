# Masters-Project — Claude Handoff Document
**Last updated:** 2026-05-16  
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
Stage 5: LLM scoring    signal/scoring/llm_scoring.py                 ✓ COMPLETE (v8, $1.09)
           ↓  paragraphs_llm_scored.csv, monthly_signal_llm.csv, bvar_signal_llm.csv
Stage 5b: Few-shot      signal/scoring/llm_scoring.py --few-shot      ✓ COMPLETE (ρ=0.969)
           ↓  monthly_signal_llm_fewshot.csv
Stage 5c: Validation    signal/validation/                            ✓ COMPLETE (2026-05-13)
           ↓  72 human-labeled paragraphs; macro F1=0.831, κ=0.750, accuracy=0.833
Stage 7a: DFM              econometrics/identification/               ✓ COMPLETE
           ↓  DFM on inflation, EMAE, fiscal balance + Milei dummy (Spec2)
           ↓  2 factors extracted
Stage 7b: Shock extraction econometrics/identification/               ✓ COMPLETE
           ↓  net_hawkish_llm_z regressed on DFM factors → residuals = purged shock
Stage 7c: Local Projections econometrics/estimation/                  ✓ COMPLETE
           ↓  Jordà (2005) LP-IRFs using residual shock series
           ↓  LP spec: y_{t+h} − y_{t−1} (cumulative changes)
           ↓  Controls: f1, f2, FE presidente, inf. realizada, f_e, Δpie{t-1}
           ↓  HAC SEs (Newey-West) + 68% and 95% CI bands reported
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
| `data/interim/monthly_signal_llm.csv` | **Monthly LLM signal — primary LP input** |
| `data/interim/monthly_signal_llm_fewshot.csv` | Few-shot robustness signal (ρ=0.969 vs zero-shot) |
| `data/interim/llm_scores_checkpoint.json` | LLM scoring checkpoint (resume support) |
| `data/processed/bvar_signal.csv` | Clean dictionary signal, BVAR-ready |
| `data/processed/bvar_signal_llm.csv` | Clean LLM + dictionary signal, BVAR-ready |
| `data/processed/signals_clean.csv` | Final compiled signal (125 months; signal_main, signal_robust, signal_dictionary) |
| `signal/dictionaries/hawkish_terms.txt` | 61 hawkish terms (v6) — used in Stage 4 only |
| `signal/dictionaries/dovish_terms.txt` | 46 dovish terms (v6) — used in Stage 4 only |
| `signal/scoring/tfidf_dictionary.py` | Stage 4 scoring pipeline (v8 — keyword fiscal filter) |
| `signal/scoring/llm_scoring.py` | Stage 5 LLM paragraph scoring (Claude API) |
| `signal/topic_modeling/wordclouds.py` | Word cloud generator — reads `paragraphs_scored.csv`, filters by `is_fiscal` (v8 BBD flag) |
| `outputs/figures/wc_fiscal_<president>.png` | Fiscal-filtered word clouds (v8) — used in presentation slides |
| `outputs/figures/llm_vs_dict_signal.png` | 4-panel LLM vs dictionary validation chart — appendix slide |
| `outputs/tables/scoring_summary.txt` | Dictionary scoring audit |
| `outputs/tables/llm_scoring_summary.txt` | LLM scoring audit + cross-validation stats |
| `outputs/tables/llm_scoring_summary_fewshot.txt` | Few-shot robustness scoring summary |
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

## Signal Results

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
January 2024 (Davos): LLM z = +1.542 (correctly hawkish; dictionary z = +0.238)

### Few-Shot Robustness (Stage 5b) — COMPLETE

Re-run with 9 labeled calibration examples (3 per class, corpus-drawn, president names stripped). Spearman ρ = 0.969 vs zero-shot baseline. Score distributions shift slightly toward neutral (expected calibration effect) but monthly z-scores are essentially unchanged (max president-level diff = 0.007z). Zero-shot retained as primary result.

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

**Key divergences (v8):** January 2024 Milei gap = 1.30z (LLM +1.542 vs Dict +0.238) — LLM correctly captures Davos ideological hawkishness. Largest overall divergence: Macri 2016-05 (LLM −1.228 vs Dict +0.325, gap = 1.55z) — likely reflects gradualismo framing where fiscal vocabulary was present but stance was non-committal.

---

## Econometric Results

### Specification (Spec2 — primary)

**DFM:** Estimated on inflation (INDEC IPC), EMAE, fiscal balance. Milei dummy included in DFM to account for structural break in macro regime. Two factors extracted. EMBI+, ARS/USD, and REM expectations excluded from DFM (outcome variables).

**Shock extraction:** `net_hawkish_llm_z` regressed on DFM factors. Residuals = purged communication shock (orthogonal to prevailing macro conditions).

**LP specification:**
`y_{t+h} − y_{t−1} = αh + βh·ε̂_t + γh·controls_t + u_{t+h}`

Controls: f1, f2, FE presidente, realised inflation, f_e, Δπ_{t-1}. LHS is cumulative change. HAC standard errors (Newey-West). 68% and 95% CI bands reported.

### Headline IRF — REM Inflation Expectations (Expertos)

A +1σ purged hawkish communication shock produces a persistent negative cumulative response in 12-month expert inflation expectations:

| Horizon | β_h (pp) | t-stat |
|---------|----------|--------|
| h=1 | ~−4 | −2.7 |
| h=2 | ~−7 | −2.7 |
| h=3 | ~−12 | −2.4 |
| h=4 | ~−14 | −2.2 |
| h=5 | ~−14 | −2.1 |
| h=6–12 | ~−12 to −13 | −1.2 to −1.8 |

The 68% CI bands exclude zero through approximately h=5. The 95% bands are wider but the point estimates are stable and economically signed correctly throughout the horizon. Di Tella consumer expectations (median and high-expectation tail) show qualitatively similar responses with the high-expectation tail (Cola Alta) responding most strongly — consistent with more inflation-sensitive consumers updating more on presidential fiscal signals.

**Key finding:** Fiscal hawkish communication causally reduces inflation expectations. The effect deepens through quarter 1, stabilises around −12 to −14 pp, and is persistent through the full 12-month horizon.

---

## Fiscal Filter — Keyword Filter (v8)

**Method:** Baker, Bloom & Davis (2016) EPU keyword-in-text approach, directly following their Appendix B fiscal category methodology. A paragraph is classified as fiscal if it contains at least one of 22 keywords (stems — each matches all morphological variants via `\b<stem>\w*`).

**v8 keywords (22 total):**
`deficit, superavit, gasto, presupuest, fiscal, impuesto, deuda, inflacion, ajuste, subsidio, recaudacion, austeridad, obra publica, inversion publica, inversion social, emision, jubil, pension, salario, fmi, privatiz, pobreza`

This replaced the LDA threshold filter (v6) which had an asymmetric bias — it captured ~84% of hawkish hits but only ~40% of dovish hits. BBD's symmetric keyword approach fixes this.

**Fiscal paragraph counts by president (v8):**
- Macri: 873 fiscal paragraphs / 5,247 total (16.6%) — 17.8 fiscal paras/month
- AF: 1,140 fiscal paragraphs / 6,264 total (18.2%) — 24.3 fiscal paras/month
- Milei: 1,891 fiscal paragraphs / 4,686 total (40.4%) — 65.2 fiscal paras/month

**Milei speech note:** Milei gives fewer speeches than Macri/AF (6.4/month vs 13.4/month) but his speeches are longer and have far higher fiscal content density (35% vs 8–13%). The LLM signal level difference reflects genuine fiscal discourse intensity, not speech volume.

**Embedding filter approach (considered, rejected):** An Ash & Hansen (2023) semantic similarity filter using multilingual-E5-base embeddings was implemented and tested. It failed due to the anisotropy problem in transformer embeddings — all paragraphs scored above the highest threshold tested (0.45). The BBD keyword approach was retained as simpler, more transparent, and directly citable.

---

## LLM Scoring Methodology — Literature Grounding

### Why LLM-primary over dictionary-primary

The dictionary has a structural limitation: Milei's most important speeches (Davos Jan 2024, WEF, UN General Assembly) communicate fiscal stance through philosophical argument rather than fiscal-accounting vocabulary. The LLM evaluates fiscal *intent* from context.

**Speaker identity is never shown to the model** — no president name, speech title, or date in the prompt. The model scores content, not speaker.

### Literature basis for the approach

| Paper | Approach | Key insight for our project |
|-------|----------|----------------------------|
| Hansen & Kazinnik (2024) | Prompted GPT for FOMC sentences | Validates API-prompting (not fine-tuning) for central bank text |
| Bernoth (2025, DIW dp2137) | Fine-tuned RoBERTa, BVAR | Same EPU formula; communication shock = residual from macro regression |
| IMF (2025, WP 2025/109) | Fine-tuned LLM, sentence-level | Scale and multilingual validation of LLM classification |
| Bank of England (2025, SWP 1127) | "Tens-of-shot" embeddings | ~20–30 labeled examples sufficient; validates few-shot approach |
| Baker, Bloom & Davis (2016) | EPU keyword counting | Foundation for keyword fiscal filter and (H−D)/P aggregation formula |
| Gentzkow, Kelly & Taddy (2019) | Text-as-data survey (JEL) | Document unit choice (paragraph vs sentence) |

### Human validation — COMPLETE (2026-05-13)

72 fiscal paragraphs labeled manually (stratified: 24 per class × 3 presidents), blinded — no president name, date, or LLM score shown during labeling. ~25% of sample drawn from LLM-dictionary disagreement cases.

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

Zero extreme errors (no dovish↔hawkish misclassifications). All 12 errors are adjacent-class. LLM-dict disagreement cases perform identically to agreement cases (both 0.833 accuracy).

---

## Scoring Pipelines — Technical Notes

### Stage 4: tfidf_dictionary.py (v8)

- **Keyword fiscal filter (v8):** 22 keywords — BBD (2016) Appendix B approach
- **EPU paragraph counting:** each fiscal para casts binary vote has_hawkish / has_dovish
- **Winsorisation:** 2.5/97.5 pct before z-scoring
- **Two-tier negation:** Tier 1 strong (10-word), Tier 2 weak (3-word)
- **Direction-aware elimination:** suppresses dovish hits only
- **Negation suppression rate (v8):** Macri 2.4% H / 2.4% D; AF 0.0% H / 2.7% D; Milei 2.4% H / 10.4% D

### Stage 5: llm_scoring.py

```bash
cd ~/Desktop/Masters-Project
source .venv/bin/activate
ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring.py --dry-run
ANTHROPIC_API_KEY="sk-ant-..." python signal/scoring/llm_scoring.py

# v8 run stats: 3,904 paragraphs, 391 batches, ~$1.09, ~28 min
```

Checkpoint/resume: `data/interim/llm_scores_checkpoint.json` — delete to rescore from scratch.

---

## Known Data Notes

- **Two missing AF months:** January 2020 and November 2023 have zero fiscal paragraphs. Treated as 0 (absence of fiscal communication = neutral stance). Stated in thesis Section 3.1.
- **February 2026 anomaly:** Milei P_t=1 (single ceremonial paragraph, `deuda` false positive). Signal = −0.113z. Retained in sample; flagged as thin month in Section 3.1.
- **Dead dictionary term:** `actualizacion tarifaria` — zero hits across full corpus; noted in audit.

---

## Key Numbers to Know

- **Corpus:** 1,498 speeches, 16,197 paragraphs, 3 presidents
- **Dictionary:** 61 hawkish + 46 dovish = 107 terms (Stage 4 only — not used in LLM scoring)
- **Fiscal paragraphs scored by LLM:** 3,904 (873 Macri + 1,140 AF + 1,891 Milei) — v8 filter
- **LLM primary LP input column:** `net_hawkish_llm_z` in `data/interim/monthly_signal_llm.csv`
- **Dictionary robustness column:** `net_hawkish_z` in `data/interim/monthly_signal.csv`
- **Cross-validation:** Pearson r = 0.839, Spearman ρ = 0.835 (LLM v8 vs dictionary v8)
- **Few-shot robustness:** Spearman ρ = 0.969 vs zero-shot baseline
- **January 2024:** LLM z = +1.542, Dict z = +0.238 — Davos correctly hawkish; LLM stronger
- **President ordering (LLM v8):** Milei +1.29z > Macri +0.18z > AF −0.98z (gap = 2.27z)
- **Fiscal keyword filter:** 22 keywords v8 (symmetric — BBD 2016; rerun complete 2026-05-02)
- **LLM model used:** claude-haiku-4-5-20251001, temperature=0
- **Human validation:** accuracy=0.833, macro F1=0.831, κ=0.750; zero extreme errors
- **Headline IRF (REM, h=2):** β = ~−7pp, t = −2.7; effect peaks ~−14pp at h=4

---

## Literature Basis

- **Fiscal keyword filter:** Baker, Bloom & Davis (2016) EPU keyword-in-text approach, Appendix B fiscal category
- **Filter design / text methods survey:** Ash & Hansen (2023) *Text Algorithms in Economics* — embedding approach considered but rejected
- **Aggregation formula:** Baker, Bloom & Davis (2016) (H_t − D_t) / P_t EPU paragraph-counting formula
- **LLM scoring:** Hansen & Kazinnik (2024); Bank of England (2025) SWP 1127; IMF (2025) WP 2025/109
- **DFM identification:** Bernoth (2025, DIW dp2137) Section 5; Stock & Watson (2002) dynamic factor models
- **BVAR framework:** Bernoth (2025); Istrefi & Piloiu (2014) — retained as robustness
- **Local Projections:** Jordà (2005)
- **Dictionary basis:** Blanchard & Leigh (2013); Dornbusch & Edwards (1991); Alesina & Ardagna (2010); Kopits & Symansky (1998)
- **Document unit sensitivity:** Gentzkow, Kelly & Taddy (2019)
- **TVP-VAR:** Primiceri (2005)
