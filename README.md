# Fiscal Hawkishness and Inflation Expectations in Argentina (2015–2026)

A master's thesis project combining natural language processing of Argentine presidential speeches with Bayesian structural VAR analysis to study how fiscal-policy rhetoric affects the anchoring of inflation expectations under Kirchnerism, Macri's gradualism, and Milei's shock therapy.

## Research question

Does the fiscal-hawkishness content of presidential communication causally shift inflation expectations in Argentina, and has the pass-through from rhetoric to expectations changed under the Milei administration? The hypothesis, motivated by Reny's (2025) natural-language equilibrium framework, is that presidential speech moves expectations more strongly when the speaker's observable policy actions make rational deception implausible — a condition arguably satisfied post-December 2023 but not under prior administrations.

## Approach in one paragraph

We construct a monthly time series of fiscal hawkishness from Argentine presidential speeches (2015–2026) using a validated dictionary-based NLP pipeline, cross-checked against LLM-based scoring and human labels. We then embed this series in a Bayesian structural VAR with inflation, inflation expectations (BCRA REM), the parallel exchange rate gap, and sovereign risk, identifying the structural hawkishness shock via sign restrictions and a narrative proxy-SVAR built around pre-specified announcement dates. We test for regime change in the rhetoric-to-expectations pass-through around Milei's inauguration.

## Theoretical grounding

The econometric design follows Istrefi and Piloiu (2014), who use a news-based policy measure in a structural BVAR to study the response of long-horizon inflation expectations to policy shocks. Reny (2025) provides the theoretical motivation for when presidential language should be treated as informative versus cheap talk in a signaling game.

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

Collects the full corpus of presidential speeches, press conferences, and major televised addresses (*cadenas nacionales*) from the Casa Rosada archive, supplemented by key international appearances (Davos, UN General Assembly, CPAC). Each document is stored with metadata: date, speaker, venue, audience type, word count, and event category. The pre-registered sampling frame is documented in `signal/scraping/sampling_frame.md`.

### Stage 2. Preprocessing (`signal/preprocessing/`)

Standard Spanish-language text normalization: sentence segmentation with spaCy's `es_core_news_lg`, lowercasing, punctuation handling, removal of ceremonial boilerplate (e.g., opening salutations, applause markers). Output is a paragraph-level dataframe with one row per paragraph and one `speech_id` per document.

### Stage 3. Topic modeling — LDA (`signal/topic_modeling/`)

Before any hawkishness scoring is applied, we run Latent Dirichlet Allocation on the paragraph-level corpus to identify the latent topic structure of Argentine presidential discourse. LDA serves two roles in this pipeline, and both are important for the validity of the final score.

**Role 1 — fiscal-relevance filtering.** Presidential speeches cover many topics besides fiscal policy: security, international affairs, sports, ceremonial praise, immigration, labor. Applying a fiscal-hawkishness dictionary to the full corpus mechanically dilutes the signal, because non-fiscal paragraphs contribute zero hawkish or dovish terms and drag the speech-level mean toward zero. We use LDA to compute, for each paragraph, its posterior probability of belonging to a fiscal/monetary-policy topic (identified ex-post from the topic-word distributions). Hawkishness scoring is then run either (a) only on paragraphs whose fiscal-topic probability exceeds a threshold, or (b) on all paragraphs but weighted by their fiscal-topic probability. This produces a sharper, more interpretable score and is a defensible answer to the "is your measure just picking up more speech, not more fiscal speech?" critique.

**Role 2 — descriptive validation.** The LDA topic distributions are an interpretable, model-free decomposition of the corpus. We report (i) the top-*k* words per topic to demonstrate that a fiscal/monetary topic is actually recovered and is distinct from other policy topics, (ii) the time series of the fiscal-topic share by president, which is itself a useful descriptive statistic for the thesis, and (iii) the distribution of dictionary-flagged hawkish terms across LDA topics, which checks that hawkish terms concentrate in the fiscal topic rather than being scattered across unrelated topics. If the dictionary's hits cluster cleanly in the fiscal topic, that is independent evidence that the score measures what it claims to measure.

LDA is implemented in Gensim with a grid search over the number of topics (*k* ∈ {8, 10, 12, 15, 20}) selected by coherence score (*c_v*) on a held-out validation split. Hyperparameters (α, β) are optimized with the built-in auto-tuning.

### Stage 4. Dictionary-based scoring (`signal/scoring/tfidf_dictionary.py`)

A Spanish fiscal hawkish/dovish dictionary is hand-curated in `signal/dictionaries/` with separate lists for hawkish terms (e.g., *ajuste fiscal*, *déficit cero*, *equilibrio fiscal*, *motosierra*, *emisión monetaria cero*, *austeridad*) and dovish terms (e.g., *inversión social*, *estado presente*, *políticas redistributivas*, *obra pública*, *estímulo fiscal*). Scoring uses normalized term frequency at the paragraph level, weighted by LDA fiscal-topic probability, and aggregated to the speech level. The dictionary itself is version-controlled as part of the methodology.

### Stage 5. LLM-based scoring (`signal/scoring/llm_scoring.py`)

Paragraphs flagged as fiscal-relevant by LDA are independently scored with a large language model (via API) on a –3 to +3 hawkishness scale using a rubric-driven prompt that requires a justification and a quoted exemplar phrase. Temperature is set to 0 and the model version is pinned for reproducibility. The LLM score serves as a robustness measure, not the primary measure.

### Stage 6. Validation (`signal/validation/`)

A stratified random sample of ≈300–500 paragraphs is hand-labeled by two independent coders using the same rubric as the LLM prompt. We report:

- Inter-rater reliability (Cohen's κ, Krippendorff's α) between human coders
- Correlation between dictionary score and human labels
- Correlation between LLM score and human labels
- Correlation between dictionary score and LLM score
- Correlation of the monthly aggregate with external markers: ARS blue/CCL gap, EMBI-Argentina spread, and the BCRA REM fiscal-balance expectation

The validation table is the single most important artefact produced by this pipeline. If the correlations are weak, the rest of the thesis cannot be trusted.

### Stage 7. Monthly aggregation (`signal/scoring/aggregate_monthly.py`)

Speech-level scores are aggregated to the calendar month using word-count weights. Months with multiple major addresses are handled via weighted averaging; months with no presidential communication are interpolated using the lagged speech-level score (with a robustness check that uses zeros instead). The output is `data/processed/hawkishness_monthly.csv`.

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

Five-week write-up phase. Active priorities: validation sprint (hand-labels, IRR, external correlations), one identification upgrade beyond Cholesky (narrative proxy-SVAR preferred), and the pre-Milei / post-Milei regime comparison as the headline result.

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
