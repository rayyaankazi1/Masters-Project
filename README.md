# Fiscal Hawkishness and Inflation Expectations in Argentina (2015–2026)

A master's thesis project combining natural language processing of Argentine presidential speeches with Bayesian structural VAR analysis to study how fiscal-policy rhetoric affects the anchoring of inflation expectations under Macri, Fernandez and Milei.

## Research question

Does the fiscal-hawkishness content of presidential communication causally shift inflation expectations in Argentina, and has the pass-through from rhetoric to expectations changed under the Milei administration? The hypothesis is that presidential speech moves expectations more strongly when the speaker's observable policy actions make rational deception implausible — a condition arguably satisfied post-December 2023 but not under prior administrations.

## Approach in one paragraph

We construct a monthly time series of fiscal hawkishness from Argentine presidential speeches (2015–2026) using a validated dictionary-based NLP pipeline, cross-checked against LLM-based scoring and human labels. We then embed this series in a Bayesian structural VAR with inflation, inflation expectations (BCRA REM) and expectations of fiscal balance, identifying the structural hawkishness shock via sign restrictions and a narrative proxy-SVAR built around pre-specified announcement dates. We test for regime change in the rhetoric-to-expectations pass-through around Milei's inauguration.

## Theoretical grounding

The econometric design follows Istrefi and Piloiu (2014), who use a news-based policy measure in a structural BVAR to study the response of long-horizon inflation expectations to policy shocks. 

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
├── econometrics/               # Pipeline 2: signal → BVAR → IRFs
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

The goal of this pipeline is a single output file, `data/processed/hawkishness_monthly.csv`, with columns `date, score_dictionary, score_llm, n_speeches, n_words, fiscal_topic_share`. The econometrics pipeline consumes this file and nothing else from the signal side.

### Stage 1. Scraping (`signal/scraping/`)

Collects the full corpus of presidential speeches, press conferences, and major televised addresses (*cadenas nacionales*) from the Casa Rosada archive, ***supplemented by key international appearances (Davos, UN General Assembly, CPAC)***. Each document is stored with word count and dare. The pre-registered sampling frame is documented in `signal/scraping/sampling_frame.md`.

### Stage 2. Preprocessing (`signal/preprocessing/`)

Standard Spanish-language text normalisation: sentence segmentation with spaCy's `es_core_news_lg`, lowercasing, punctuation handling, removal of ceremonial boilerplate (e.g., opening salutations, applause markers). Output is a paragraph-level dataframe with one row per paragraph and one `speech_id` per document.

### Stage 3. Topic modeling — LDA (`signal/topic_modeling/`)

LDA is run on the paragraph-level corpus to characterise the latent topic structure of Argentine presidential discourse. In the current pipeline, LDA serves a **descriptive and validation role only** — it is not the load-bearing fiscal filter applied before scoring (that role is handled by a keyword filter in Stage 4, see below). Retaining LDA in the pipeline is motivated by two things.

**Descriptive characterisation.** We report (i) the top-*k* words per topic to demonstrate that a fiscal/monetary topic is recovered and is distinct from other policy topics, (ii) the time series of the fiscal-topic share by president as a model-free complement to the dictionary signal, and (iii) the distribution of dictionary-flagged hawkish terms across LDA topics, which checks that hawkish hits concentrate in the fiscal topic rather than scattering across unrelated ones.

**Mutual validation.** If the LDA-derived fiscal-topic share and the dictionary/LLM hawkishness signal correlate strongly at monthly frequency, this constitutes independent model-free evidence that both are measuring the same underlying construct. We optionally fit a Dynamic Linear Model (DLM) on the monthly fiscal-topic proportions (Battaglia & Salunina 2020) and compare the Kalman-smoothed latent factor with `net_hawkish_z` as a further robustness check.

LDA is implemented in Gensim with a grid search over the number of topics (*k* ∈ {5, 8, 10, 12, 15, 20}) selected by coherence score (*c_v*) on a held-out validation split. Hyperparameters (α, β) are optimized with the built-in auto-tuning.

### Stage 4. Dictionary-based scoring (`signal/scoring/tfidf_dictionary.py`)

A Spanish fiscal hawkish/dovish dictionary is hand-curated in `signal/dictionaries/` with 61 hawkish terms (e.g., *ajuste fiscal*, *déficit cero*, *equilibrio fiscal*, *emisión monetaria cero*, *superávit fiscal*) and 46 dovish terms (e.g., *obra pública*, *estado presente*, *salario real*, *asignación universal*, *tarjeta alimentar*). Paragraphs are identified as fiscal-relevant using a **keyword filter (v8)** following the Baker, Bloom & Davis (2016) EPU keyword-in-text methodology (Appendix B fiscal category): a paragraph is included if it contains at least one of 22 core fiscal/monetary keywords. Each keyword is a stem matched with a `\b<stem>\w*` pattern to capture all morphological variants. The 22 keywords cover both hawkish fiscal-accounting vocabulary (*déficit*, *fiscal*, *presupuesto*, *ajuste*, *austeridad*, *emisión*) and dovish social-spending vocabulary (*obra pública*, *jubilados*, *pensión*, *salario*, *pobreza*), making the filter symmetric across policy directions. An earlier LDA-threshold filter (v6) was replaced because it captured ~84% of hawkish hits but only ~40% of dovish hits due to the different topic distributions of fiscal-accounting vs. social-spending vocabulary.

Scoring follows the **EPU paragraph-counting methodology** (Baker, Bloom & Davis 2016): each fiscal paragraph casts a binary vote (has\_hawkish / has\_dovish), and the monthly signal is `(H_t − D_t) / P_t` where H_t and D_t are hawkish- and dovish-hit fiscal paragraph counts and P_t is total fiscal paragraphs in the month. The signal is winsorised at 2.5/97.5 percentiles before z-scoring over the full cross-president sample. A two-tier negation filter (Tier 1: 10-word window for critical/ironic framing; Tier 2: 3-word window for bare negation) suppresses false hits at an overall rate of ~3.4%.

Following BBD's validation procedure, the keyword filter is audited against ~150 human-labeled paragraphs to report precision and recall. BBD validate their newspaper filter by human-reading 12,000 articles; our smaller audit serves the same methodological function.

An embedding-based semantic filter (Ash & Hansen 2023) using multilingual-E5-base cosine similarity was considered as an alternative but rejected: the anisotropy problem in transformer embedding spaces meant all 16,197 paragraphs scored above the highest tested threshold (0.45), making the model unable to discriminate fiscal from non-fiscal text without fine-tuning on labeled pairs. The keyword approach was retained as more transparent and directly replicable.

**Dictionary signal (`net_hawkish_z` in `data/interim/monthly_signal.csv`) — robustness series:**

| President | N months | Mean-z | Std-z |
|-----------|----------|--------|-------|
| Macri (2015–2019) | 49 | +0.25 | 0.88 |
| AF (2019–2023) | 47 | −0.91 | 0.86 |
| Milei (2023–2026) | 29 | +1.06 | 0.54 |

Milei–AF separation: 1.97 z-units. Numbers above are from v7 filter; v8 rerun pending.

### Stage 5. LLM-based scoring (`signal/scoring/llm_scoring.py`)

Fiscal paragraphs (same keyword filter as Stage 4) are independently scored with a large language model (Claude API) using a **zero-shot blind rubric**: the model assigns −1 (dovish), 0 (neutral), or +1 (hawkish) based solely on fiscal content, with no president name, speech title, or date in the prompt. Temperature is set to 0 and the model version is pinned for reproducibility. The LLM and dictionary pipelines are fully independent — the LLM does not use the hawkish/dovish dictionaries.

The LLM approach captures fiscal intent communicated through ideological argument rather than fiscal-accounting vocabulary. For example, Milei's Davos 2024 speech argues that state financing via taxes is coercive and that monetary emission causes poverty — content the dictionary misses entirely but which the LLM scores +1 hawkish. This is the **primary BVAR signal**; the dictionary signal (`net_hawkish_z`) is retained as a robustness/replication check.

**LLM signal (`net_hawkish_llm_z` in `data/processed/bvar_signal_llm.csv`) — primary series:**

| President | N months | Mean-z | Std-z |
|-----------|----------|--------|-------|
| Macri (2015–2019) | 49 | +0.38 | 0.87 |
| AF (2019–2023) | 47 | −1.03 | 0.82 |
| Milei (2023–2026) | 29 | +1.03 | 0.59 |

Milei–AF separation: 2.06 z-units. Cross-validation with dictionary signal: Pearson r = 0.767, Spearman ρ = 0.765 (monthly z-scores, n = 125). January 2024 correction: dictionary z = −1.532 → LLM z = +1.542 (Davos ideological arguments correctly scored hawkish).

**Planned robustness (Stage 5b):** re-run with few-shot prompting (9 labeled Spanish examples covering all three presidents and edge cases); compare Spearman ρ against zero-shot baseline. Human validation holdout: ~60 manually labeled paragraphs, precision/recall/F1 reported.

### Stage 6. Validation (`signal/validation/`)

A stratified random sample of ≈300–500 paragraphs is hand-labeled by two independent coders using the same rubric as the LLM prompt. We report:

- Inter-rater reliability (Cohen's κ, Krippendorff's α) between human coders
- Correlation between dictionary score and human labels
- Correlation between LLM score and human labels
- Correlation between dictionary score and LLM score
- Correlation of the monthly aggregate with external markers: ARS blue/CCL gap, EMBI-Argentina spread, and the BCRA REM fiscal-balance expectation

The validation table is the single most important artefact produced by this pipeline. If the correlations are weak, the rest of the thesis cannot be trusted.

### Stage 7. Monthly aggregation (integrated into `signal/scoring/tfidf_dictionary.py`)

Monthly aggregation uses EPU-style pooling: H_t, D_t, P_t are summed across all speeches in the month before dividing, giving equal weight to every fiscal paragraph regardless of which speech it came from. A robustness column `net_hawkish_rob_z` uses total paragraphs N_t as denominator. The primary BVAR-ready output is `data/processed/bvar_signal.csv` (columns: `year_month`, `president`, `net_hawkish_z`). Full signal detail (H_t, D_t, P_t, raw signal, robustness column) is in `data/interim/monthly_signal.csv`.

## Pipeline 2 — Econometrics (`econometrics/`)

### Stage 1. Data preparation (`econometrics/data_prep/`)

Merges the monthly hawkishness signal with macro data: inflation (INDEC IPC Nacional, with pre-2017 caveats noted), REM inflation expectations (12m and 24m horizons, plus cross-sectional dispersion as an anchoring proxy), the ARS parallel-market premium, EMBI-Argentina, the monetary policy rate, and a fiscal-balance series. All series run at monthly frequency over 2015m1–2026m2.

### Stage 2. Estimation (`econometrics/estimation/`)

Bayesian VAR estimated in R using the `bsvars` / `BVAR` package with Minnesota/Litterman priors, following the specification in Istrefi and Piloiu (2014). Lag length is selected by BIC with DIC as a cross-check; baseline specification uses *p* = 4 lags on monthly data.

### Stage 3. Identification (`econometrics/identification/`)

Three identification schemes are implemented and compared:

1. **Cholesky ordering** (baseline, following Istrefi–Piloiu): hawkishness ordered before macro variables, assuming no contemporaneous feedback from inflation expectations to rhetoric.
2. **Sign restrictions**: a hawkish shock is required to raise the hawkishness score on impact, weakly appreciate the parallel FX, and weakly lower country risk, while leaving inflation expectations free to respond in either direction.
3. **Narrative proxy-SVAR** (Mertens–Ravn 2013, Stock–Watson 2018, Gertler–Karadi 2015): an external instrument is constructed from pre-registered announcement dates (DNU 70/2023, the December 2023 devaluation announcement, Ley Bases votes, major presidential vetoes, inauguration addresses, Davos/CPAC appearances). The event-day surprise is measured using the residualized hawkishness score and, where intraday data permit, the high-frequency change in ARS CCL and EMBI spreads. Event-level surprises are summed within each calendar month to produce the monthly instrument, with most entries at zero and nonzero spikes on event months. Weak-IV tests (Olea–Pflueger) are reported.

### Stage 4. Results (`econometrics/results/`)

Produces impulse response functions, forecast error variance decompositions, and historical decompositions. The headline exhibit is the **pre-Milei vs. post-Milei split-sample comparison of IRFs**: the response of 24-month-ahead REM inflation expectations (and its cross-sectional dispersion) to a one-standard-deviation hawkishness shock, estimated separately on the pre-December 2023 subsample and on the Milei subsample. Structural break tests (Bai–Perron on the hawkishness series; Chow and QLR on reduced-form VAR coefficients) provide formal evidence on whether the regime change is statistically detectable.

### Stage 5. Robustness (`econometrics/robustness/`)

Dictionary vs. LLM score in the BVAR, alternative variable orderings, lag length robustness, subsample stability, alternative monthly aggregation rules (word-count vs. equal-weight), and alternative inflation-expectation measures (UTDT household survey as a non-professional complement).

## Data sources

- **Presidential speeches**: Casa Rosada online archive; YouTube transcripts for televised addresses; official transcripts for Davos, CPAC, and UN appearances.
- **Inflation**: INDEC IPC Nacional (2017m1 onwards); City of Buenos Aires IPCBA for earlier periods; academic CPI reconstructions (Cavallo 2013) for the 2007–2015 INDEC-manipulation episode.
- **Inflation expectations**: BCRA REM — *Relevamiento de Expectativas de Mercado*; UTDT Survey of Household Expectations.
- **Financial markets**: BCRA and Rava for ARS official/CCL/blue; JPMorgan EMBI Argentina; BYMA for sovereign CDS.
- **Fiscal balance**: Ministry of Economy monthly primary balance series.

Full retrieval scripts and source URLs are in `data/README.md`.

## Reproducibility

Python dependencies are pinned in `requirements.txt` (signal pipeline). R dependencies are managed via `renv` inside `econometrics/` (`renv.lock`). LLM API calls use `temperature=0` and pinned model versions; the exact model string is recorded in the run logs produced by `signal/scoring/llm_scoring.py`. Random seeds are set in all stochastic steps (LDA, BVAR sampling). Raw data is not redistributed but retrieval scripts are included.

## Timeline and status

**Signal pipeline (Stages 1–5): COMPLETE as of 2026-04-30.**  
Primary BVAR input `net_hawkish_llm_z` is in `data/processed/bvar_signal_llm.csv`, covering 125 monthly observations with valid signal (2015-12 → 2026-04; 2 AF months NaN due to zero fiscal paragraphs). Dictionary robustness series `net_hawkish_z` is in `data/interim/monthly_signal.csv`. Cross-validation: Pearson r = 0.767, Spearman ρ = 0.765.

**Active priorities:** (1) Re-run Stage 4 (`tfidf_dictionary.py`) and Stage 5 (`llm_scoring.py`) with the v8 keyword filter. (2) BBD keyword audit + human validation holdout — label ~150 paragraphs (fiscal/non-fiscal and hawkish/neutral/dovish), report filter precision/recall and LLM F1 in one pass. (3) Stage 5b few-shot robustness run — add 9 labeled examples to prompt, compare ρ against zero-shot baseline. (4) External validation — correlate `net_hawkish_llm_z` with Argentine primary balance, EMBI+, ARS/USD. (5) BVAR construction in `econometrics/` (currently empty skeleton), using TVP-VAR (Primiceri 2005) as preferred specification to accommodate the structural break across administrations.

## References

Istrefi, K., & Piloiu, A. (2014). *Economic Policy Uncertainty and Inflation Expectations*. Banque de France Working Paper 511.

Reny, P. J. (2025). *Natural Language Equilibrium: Off-Path Conventions I*. Working paper, University of Chicago.

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics* 131(4).

Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. *American Economic Review* 103(4).

Gertler, M., & Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. *American Economic Journal: Macroeconomics* 7(1).

Stock, J. H., & Watson, M. W. (2018). Identification and estimation of dynamic causal effects in macroeconomics using external instruments. *Economic Journal* 128(610).

Apel, M., & Blix Grimaldi, M. (2014). How informative are central bank minutes? *Review of Economics* 65(1).

Hansen, S., & McMahon, M. (2016). Shocking language: understanding the macroeconomic effects of central bank communication. *Journal of International Economics* 99.

Shapiro, A. H., & Wilson, D. J. (2019). Taking the Fed at its word: a new approach to estimating central bank objectives using text analysis. FRBSF Working Paper 2019-02.

Bernoth, K. (2025). Dovish coos or hawkish screech? Measuring ECB communication using large language models. *DIW Berlin Discussion Paper* 2137.

Hansen, S., & Kazinnik, S. (2024). Can large language models extract information usable in financial markets? *Journal of Finance* (forthcoming).

Bank of England (2025). *Using large language models to measure monetary policy communication.* Staff Working Paper 1127.

Ash, E., & Hansen, S. (2023). Text algorithms in economics. *Annual Review of Economics* 15, 659-688. [Embedding-based semantic filter approach; considered but rejected due to transformer embedding anisotropy.]

Battaglia, F., & Salunina, M. (2020). LDA topic modelling and dynamic linear models for fiscal policy uncertainty. *Working paper*.

Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Review of Economic Studies* 72(3).
