# Fiscal Hawkishness and Inflation Expectations in Argentina (2015–2026)

A master's thesis combining natural language processing of Argentine presidential speeches with a two-stage econometric identification strategy — Dynamic Factor Model purging followed by Local Projections — to study how fiscal communication causally affects inflation expectations under Macri, Fernández, and Milei.

## Research question

Does the fiscal-hawkishness content of presidential communication causally shift inflation expectations in Argentina, and has the pass-through from rhetoric to expectations changed under the Milei administration?

## Approach in one paragraph

We construct a monthly fiscal hawkishness index from Argentine presidential speeches (2015–2026) by scoring 3,904 fiscal paragraphs with a large language model (Claude Haiku, zero-shot blind rubric), aggregated via the Baker, Bloom & Davis (2016) EPU methodology. The signal is validated against an independent dictionary measure (Spearman ρ = 0.835), human labels (macro F1 = 0.831, κ = 0.750, n = 72), and a few-shot robustness run (ρ = 0.969 vs zero-shot). To address endogeneity — presidents respond to economic conditions, so the raw signal conflates genuine communication surprises with endogenous responses — we fit a Dynamic Factor Model to core macro variables (inflation, EMAE, fiscal balance) and use the residuals from a regression of the index on the estimated factors as a purged communication shock (Bernoth 2025). These residuals are passed to Local Projections (Jordà 2005); a wild bootstrap covers the full two-stage procedure. A Structural BVAR is retained as a robustness check.

## Theoretical grounding

The identification strategy follows Bernoth (2025), who constructs ECB communication shocks as residuals from a regression of the stance indicator on macro-financial variables — the same logic applied here to purge the hawkishness index of its endogenous component. Local Projections follow Jordà (2005); the generated-regressor problem from the two-stage procedure is handled via wild bootstrap covering both stages. The BVAR robustness design draws on Istrefi and Piloiu (2014), who use a news-based policy measure in a structural BVAR to study the response of long-horizon inflation expectations to policy shocks.

## Repository structure

```
Masters-Project/
├── README.md                   # this file
├── .gitignore
├── requirements.txt            # Python dependencies for the signal pipeline
├── data/
│   ├── raw/                    # source speeches and macro data (not tracked)
│   ├── interim/                # cleaned corpora, intermediate artefacts
│   └── processed/              # final analysis-ready datasets
├── signal/                     # Pipeline 1: text → monthly hawkishness score
│   ├── scraping/
│   ├── preprocessing/
│   ├── topic_modeling/         # LDA
│   ├── dictionaries/
│   ├── scoring/
│   └── validation/
├── econometrics/               # Pipeline 2: signal → DFM → LP → IRFs
│   ├── data_prep/
│   ├── estimation/
│   ├── identification/
│   ├── results/
│   └── robustness/
├── notebooks/                  # exploratory analysis
├── outputs/
│   ├── figures/
│   └── tables/
└── paper/                      # LaTeX source
```

## Pipeline 1 — Signal construction (`signal/`)

The goal of this pipeline is a single output file, `data/processed/signals_clean.csv`, with columns `date, signal_main, signal_robust, signal_dictionary`. The econometrics pipeline consumes this file and nothing else from the signal side.

### Stage 1. Scraping (`signal/scraping/`)

Collects the full corpus of presidential speeches, press conferences, and major televised addresses (*cadenas nacionales*) from the Casa Rosada archive, supplemented by key international appearances (Davos, UN General Assembly, CPAC). Each document is stored with word count and date.

### Stage 2. Preprocessing (`signal/preprocessing/`)

Standard Spanish-language text normalisation: sentence segmentation with spaCy's `es_core_news_lg`, lowercasing, punctuation handling, removal of ceremonial boilerplate. Output is a paragraph-level dataframe with one row per paragraph and one `speech_id` per document.

### Stage 3. Topic modeling — LDA (`signal/topic_modeling/`)

LDA is run on the paragraph-level corpus to characterise the latent topic structure of Argentine presidential discourse. LDA serves a **descriptive and validation role only** — it is not the load-bearing fiscal filter (that role is handled by a keyword filter in Stage 4). It is retained to demonstrate that a fiscal/monetary topic is recovered and is distinct from other policy topics, and to provide a model-free complement to the dictionary signal.

Word cloud visualisations (`signal/topic_modeling/wordclouds.py`) are produced from the fiscal-filtered paragraph subset (paragraphs where `is_fiscal == True` in the v8 BBD keyword flag), not from an LDA topic-probability threshold. Outputs: `outputs/figures/wc_fiscal_<president>.png`.

### Stage 4. Dictionary-based scoring (`signal/scoring/tfidf_dictionary.py`)

A Spanish fiscal hawkish/dovish dictionary is hand-curated in `signal/dictionaries/` with 61 hawkish terms and 46 dovish terms. Paragraphs are identified as fiscal-relevant using a **keyword filter (v8)** following the Baker, Bloom & Davis (2016) EPU keyword-in-text methodology: a paragraph is included if it contains at least one of 22 core fiscal/monetary keyword stems. The 22 keywords cover both hawkish fiscal-accounting vocabulary (*déficit*, *fiscal*, *ajuste*, *emisión*) and dovish social-spending vocabulary (*jubilados*, *salario*, *pobreza*), making the filter symmetric across policy directions.

Scoring follows the EPU paragraph-counting methodology: each fiscal paragraph casts a binary vote (has\_hawkish / has\_dovish), and the monthly signal is `(H_t − D_t) / P_t`. The signal is winsorised at 2.5/97.5 percentiles before z-scoring over the full cross-president sample. A two-tier negation filter suppresses false hits at an overall rate of ~3.4%.

An embedding-based semantic filter (Ash & Hansen 2023) was considered as an alternative but rejected: the anisotropy problem in transformer embedding spaces meant all 16,197 paragraphs scored above the highest tested threshold (0.45). The keyword approach was retained as more transparent and directly replicable.

**Dictionary signal (`net_hawkish_z`) — robustness series:**

| President | N months | Mean-z | Std-z |
|-----------|----------|--------|-------|
| Macri (2015–2019) | 49 | +0.212 | 0.460 |
| AF (2019–2023) | 47 | −0.979 | 0.579 |
| Milei (2023–2026) | 29 | +1.228 | 0.514 |

### Stage 5. LLM-based scoring (`signal/scoring/llm_scoring.py`)

Fiscal paragraphs (same keyword filter as Stage 4) are independently scored with a large language model (Claude API) using a **zero-shot blind rubric**: the model assigns −1 (dovish), 0 (neutral), or +1 (hawkish) based solely on fiscal content, with no president name, speech title, or date in the prompt. Temperature is set to 0 and the model version is pinned for reproducibility. The LLM and dictionary pipelines are fully independent.

The LLM approach captures fiscal intent communicated through ideological argument rather than fiscal-accounting vocabulary. Milei's Davos 2024 speech — which argues that state financing via taxes is coercive and monetary emission causes poverty — is scored +1 hawkish by the LLM (z = +1.542) but largely missed by the dictionary (z = +0.238). This is the **primary LP signal**; the dictionary signal is retained as a robustness check.

**LLM signal (`net_hawkish_llm_z`) — primary series:**

| President | N months | Mean-z | Std-z | H% / N% / D% |
|-----------|----------|--------|-------|---------------|
| Macri (2015–2019) | 49 | +0.179 | 0.622 | 28.8 / 51.6 / 19.6 |
| AF (2019–2023) | 47 | −0.983 | 0.382 | 7.1 / 41.5 / 51.4 |
| Milei (2023–2026) | 29 | +1.291 | 0.346 | 72.9 / 26.7 / 0.4 |

Milei–AF separation: 2.27 z-units. Cross-validation with dictionary signal: Pearson r = 0.839, Spearman ρ = 0.835 (n = 123 months). Model: claude-haiku-4-5-20251001, temperature=0, cost $1.09.

**Few-shot robustness (Stage 5b):** Re-run with 9 labeled calibration examples (3 per class). Spearman ρ = 0.969 vs zero-shot baseline; max president-level z-score difference = 0.007z. Zero-shot retained as primary result.

### Stage 6. Validation (`signal/validation/`)

72 fiscal paragraphs labeled manually (stratified: 24 per class × 3 presidents), blinded — no president name, date, or LLM score shown during labeling. ~25% of sample drawn from LLM-dictionary disagreement cases.

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.833 |
| Macro F1 | 0.831 |
| Cohen's κ | 0.750 |

Per-class F1: dovish 0.857 / neutral 0.739 / hawkish 0.898. Zero extreme errors (no dovish↔hawkish misclassifications). All errors are adjacent-class. By president: Macri κ = 0.737 / AF κ = 0.690 / Milei κ = 0.616. Validation figures in `outputs/figures/validation_*.png`.

## Pipeline 2 — Econometrics (`econometrics/`)

### Stage 1. Data preparation (`econometrics/data_prep/`)

Merges the monthly hawkishness signal with macro data: inflation (INDEC IPC Nacional), REM inflation expectations (12m and 24m horizons), UTDT household expectations (Di Tella), ARS parallel-market premium, EMBI-Argentina, monetary policy rate, and fiscal balance. All series run at monthly frequency over 2015m1–2026m2.

### Stage 2. Identification — DFM + Residual Shock (`econometrics/identification/`)

**Stage 2a — Dynamic Factor Model.** A DFM is estimated on monthly inflation (INDEC IPC Nacional), EMAE, and fiscal balance, with a Milei dummy to account for the structural break in the macro regime (Spec2). Variables that are themselves outcomes of interest — EMBI+, ARS/USD, REM inflation expectations — are excluded from the DFM to avoid over-purging the signal. Two factors are extracted.

**Stage 2b — Shock extraction.** `net_hawkish_llm_z` is regressed on the DFM factors. The residuals constitute the purged fiscal communication shock: the component of presidential rhetoric orthogonal to prevailing economic conditions. This follows Bernoth (2025, Section 5).

### Stage 3. Estimation — Local Projections (`econometrics/estimation/`)

For each outcome variable y and horizon h = 0, 1, ..., 12:

`y_{t+h} − y_{t−1} = αh + βh·ε̂_t + γh·controls_t + u_{t+h}`

where ε̂_t are the DFM residuals. The LHS is the cumulative change from t−1 to t+h. Controls: f1, f2, FE presidente, realised inflation, f_e, Δπ_{t-1}. HAC standard errors (Newey-West). 68% and 95% CI bands reported. Implemented in R using the `lpirfs` package.

### Stage 4. Results (`econometrics/results/`)

**Headline result — REM expert inflation expectations:** A +1σ purged hawkish communication shock produces a persistent negative cumulative response. The effect reaches approximately −7pp at h=2 (t=−2.7), peaks around −14pp at h=4 (t=−2.2), and stabilises at −12 to −13pp through h=12. The 68% CI bands exclude zero through h=5. Di Tella consumer expectations show qualitatively similar responses, with the high-expectation tail (Cola Alta) responding most strongly.

### Stage 5. Robustness (`econometrics/robustness/`)

Dictionary vs. LLM signal in LP, alternative lag lengths, subsample stability, alternative monthly aggregation rules, alternative inflation-expectation measures, and LP using the raw signal without DFM purging. BVAR with Cholesky identification retained as an additional robustness frame.

## Data sources

- **Presidential speeches**: Casa Rosada online archive; YouTube transcripts for televised addresses; official transcripts for Davos, CPAC, and UN appearances.
- **Inflation**: INDEC IPC Nacional (2017m1 onwards); City of Buenos Aires IPCBA for earlier periods.
- **Inflation expectations**: BCRA REM (*Relevamiento de Expectativas de Mercado*); UTDT Di Tella Survey of Household Expectations.
- **Financial markets**: BCRA and Rava for ARS official/CCL/blue; JPMorgan EMBI Argentina.
- **Fiscal balance**: Ministry of Economy monthly primary balance series.

Full retrieval scripts and source URLs are in `data/README.md`.

## Reproducibility

Python dependencies are pinned in `requirements.txt` (signal pipeline). R dependencies are managed via `renv` inside `econometrics/` (`renv.lock`). LLM API calls use `temperature=0` and pinned model versions; the exact model string is recorded in the run logs. Random seeds are set in all stochastic steps (LDA, BVAR sampling). Raw data is not redistributed but retrieval scripts are included.

## References

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics* 131(4).

Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1), 161–182.

Bernoth, K. (2025). Dovish coos or hawkish screech? Measuring ECB communication using large language models. *DIW Berlin Discussion Paper* 2137.

Hansen, S., & Kazinnik, S. (2024). Can large language models extract information usable in financial markets? *Journal of Finance* (forthcoming).

Bank of England (2025). *Using large language models to measure monetary policy communication.* Staff Working Paper 1127.

IMF (2025). *Large language models and central bank communication.* Working Paper 2025/109.

Ash, E., & Hansen, S. (2023). Text algorithms in economics. *Annual Review of Economics* 15, 659–688.

Istrefi, K., & Piloiu, A. (2014). *Economic Policy Uncertainty and Inflation Expectations*. Banque de France Working Paper 511.

Stock, J. H., & Watson, M. W. (2018). Identification and estimation of dynamic causal effects in macroeconomics using external instruments. *Economic Journal* 128(610).

Hansen, S., & McMahon, M. (2016). Shocking language: understanding the macroeconomic effects of central bank communication. *Journal of International Economics* 99.

Shapiro, A. H., & Wilson, D. J. (2019). Taking the Fed at its word: a new approach to estimating central bank objectives using text analysis. FRBSF Working Paper 2019-02.

Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. *American Economic Review* 103(4).

Gertler, M., & Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. *American Economic Journal: Macroeconomics* 7(1).

Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Review of Economic Studies* 72(3).

Gentzkow, M., Kelly, B., & Taddy, M. (2019). Text as data. *Journal of Economic Literature* 57(3), 535–574.

Apel, M., & Blix Grimaldi, M. (2014). How informative are central bank minutes? *Review of Economics* 65(1).
