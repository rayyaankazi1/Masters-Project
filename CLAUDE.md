# Masters-Project — Claude Handoff Document
**Last updated:** 2026-04-27  
**Author:** Rayyaan Kazi (BSE Master's student)  
**Project:** Fiscal hawkishness NLP signal from Argentine presidential speeches as input to a Bayesian Structural VAR

---

## Project Goal

Construct a monthly fiscal hawkishness signal (`net_tf_fwsum_z`) from Argentine presidential speeches (Macri 2015–2019, Alberto Fernández 2019–2023, Milei 2023–2026) and use it as an identified shock series in a BVAR to estimate macroeconomic responses to fiscal stance changes.

---

## Pipeline Overview

```
Stage 1: Scraping       signal/scraping/scraper.py
           ↓
Stage 2: Raw corpus     data/raw/speeches_raw.csv
           ↓
Stage 3: LDA            signal/topic_modeling/lda.py
           ↓  paragraphs_lda.csv (fiscal_topic_prob per paragraph)
Stage 4: Dictionary     signal/scoring/tfidf_dictionary.py   ← CURRENT STAGE
           ↓  paragraphs_scored.csv, speeches_scored.csv, monthly_signal.csv
Stage 5: LLM scoring    (not yet built — for cross-validation)
           ↓
Stage 6: External valid (not yet built — correlate with primary balance, EMBI+)
           ↓
Stage 7: BVAR           econometrics/ (not yet built)
```

---

## Key Files

| File | Description |
|------|-------------|
| `data/raw/speeches_raw.csv` | Raw speech corpus |
| `data/interim/paragraphs_lda.csv` | Paragraph-level with LDA topic probabilities |
| `data/interim/paragraphs_scored.csv` | Paragraph-level with hawkish/dovish hit counts |
| `data/interim/speeches_scored.csv` | Speech-level aggregated scores |
| `data/interim/monthly_signal.csv` | **Primary BVAR input** — monthly signal, 127 rows |
| `signal/dictionaries/hawkish_terms.txt` | 61 hawkish terms (v5) |
| `signal/dictionaries/dovish_terms.txt` | 46 dovish terms (v5) |
| `signal/scoring/tfidf_dictionary.py` | Main scoring pipeline (v5) |
| `outputs/tables/scoring_summary.txt` | Last run summary statistics |

---

## Monthly Signal — Current State

**Primary BVAR column:** `net_tf_fwsum_z` in `data/interim/monthly_signal.csv`

**Method C (FW-sum weighted):** `monthly_score = Σ(net_tf_weighted × fiscal_weight_sum) / Σ(fiscal_weight_sum)`  
Soft fiscal filter — non-fiscal speeches contribute near-zero weight without hard exclusion.

**Z-score results (cross-president normalisation):**

| President | N months | Mean-z | Std-z | Interpretation |
|-----------|----------|--------|-------|----------------|
| Macri | 49 | −0.077 | 0.576 | Near-neutral (gradualismo) |
| AF | 49 | −0.647 | 0.647 | Dovish (social spending, pandemic relief) |
| Milei | 29 | +1.224 | 0.959 | Strongly hawkish (shock therapy) |

Separation: Milei–AF gap = 1.87 z-units. Ordering makes strong economic sense.

---

## Scoring Pipeline — tfidf_dictionary.py (v5)

### Key design choices
- **Morphological matching:** each term gets `\w*` suffix so "ajuste fiscal" matches "ajustes fiscales" etc.
- **Sentence-level scoring** with `finditer` — matches within sentences, not across them
- **Dual LDA weighting:** `max(fiscal_topic_prob, topic_8_prob) × n_tokens` where topic 8 = monetary
- **Two-tier negation detection:**
  - Tier 1 strong (10-word window): supuest*, llamad*, rechaz*, critic*, elimin*, en contra, opuest*, en nombre de, lo que llaman, mal llamad*, negar*, combat*, fals*, ideolog*
  - Tier 2 weak (3-word window): no, nunca, jamas
- **Direction-aware elimin* (v5):** `elimin*` only suppresses dovish hits, not hawkish. This prevented 11 false suppressions (Milei hawkish — "eliminar el impuesto inflacionario", "para eliminar la inflación, mantener el superávit fiscal")

### Negation audit results (2026-04-27)
- Total suppressed: 60 hits (34 hawkish, 26 dovish) out of ~1,750 gross
- Suppression rates: Macri 2.0%/2.8%, AF 0.0%/2.1%, Milei 2.4%/5.7% — balanced, no anomalies
- Estimated false suppressions: ~22 hits (1.3% of gross) — within acceptable threshold
- Documented residual false suppressions (not coded around, <0.7% gross each):
  - "no solo X" additive framing (4 hits)
  - "no hay plata" self-negation — term starts with "no" (3 hits, Milei)
  - Historical "nunca tuvimos X" contrast-framing (4 hits, Milei)

### Hit rate by president (fiscal paragraphs, threshold > 0.15)
- Macri: ~18% hit rate (vocabulary gap was main issue — fixed with verb-form additions)
- AF: 15.7% hit rate (genuine structural feature — AF discourse is more narrative/descriptive than signaling; much content is non-directional IMF/COVID description)
- Milei: high hit rate (strong explicit fiscal vocabulary)

---

## Dictionary — Current State

### Hawkish terms (61 terms) — `signal/dictionaries/hawkish_terms.txt`
Key sections: core fiscal tightening, credibility/rules, tax reform, Milei-era specific, monetary tightening, structural reform/liberalisation, Macri-era gradualismo vocabulary (bajar la inflacion, reducir la inflacion, ordenar las cuentas, sinceramiento, actualizacion tarifaria), verb/stem forms.

**REMOVED (contested/president-specific):** motosierra, licuadora, la casta, casta politica, viva la libertad — Milei political slogans (93–100% Milei-specific, 485 hits, no fiscal semantic content across presidents). Professors confirmed these should be excluded.

**One zero-hit term:** `actualizacion tarifaria` — Macri uses verb forms ("actualizar tarifas") not captured. Documented, left in for completeness.

### Dovish terms (46 terms) — `signal/dictionaries/dovish_terms.txt`
Key sections: expansionary fiscal stance, social spending/redistribution, AF-era specific, Macri-era dovish framing, noun forms.

**Recently added:** `obra publica` (public works spending — 16 hits in AF zero-hit fiscal paragraphs, common Argentine political phrase not covered by existing "inversion publica"), `asignacion universal` (AUH child allowance, 16 hits in zero-hit Macri fiscal paragraphs), `proceso gradual` (Macri gradualismo framing).

**CHANGED:** `subsidiar` → `subsidios` (noun form to avoid false-positive match on "subsidiaria/subsidiariedad" — 5 of 11 subsidiar* hits were false positives for subsidiary companies).

**REMOVED (false positives from Milei critical/ironic usage):** justicia social, redistribucion, redistribuir, intervencion del estado, gradualismo, acreedores, acuerdo con el fondo.

---

## What Has Been Done This Session

1. **Stage 4 pipeline fully wired** — `run()` calls `aggregate_to_monthly()`, saves `monthly_signal.csv`, adds monthly stats to `scoring_summary.txt`.

2. **Method C monthly aggregation** — FW-sum weighted (primary BVAR input), Method A equal-weight retained as robustness column. Both z-score normalised cross-president.

3. **EDA notebook Section 8** — `notebooks/02_eda_signals.ipynb` has 7 new cells covering monthly_signal.csv: Method A vs C comparison, BVAR z-score time series (raw + 3-month rolling), hawkish/dovish components area chart, fiscal coverage, summary stats.

4. **Contested terms audit** — Removed 5 Milei political slogans (motosierra, licuadora, la casta, casta politica, viva la libertad) that were 93–100% Milei-specific with no cross-president fiscal validity. Signal separation preserved at ~1.9 z-units.

5. **Stem expansion audit** — Verified `\w*` morphological expansions cause 0 false positives (barring subsidiar → subsidiaria, already fixed).

6. **Macri vocabulary gap analysis** — Found 89% zero-hit rate in Macri fiscal paragraphs was a verb-vs-noun-phrase gap. Added 5 hawkish verb-form terms (bajar la inflacion, reducir la inflacion, ordenar las cuentas, sinceramiento, actualizacion tarifaria). Macri hits increased 69→99 (+43%).

7. **Negation audit (2026-04-27)** — Full audit of 71 suppressed hits. Found 4 systematic false-suppression patterns. Fixed the most impactful: `elimin*` is now direction-aware (only suppresses dovish hits). Milei hawkish accepted hits: 1,279→1,290. Z-scores unchanged (<0.01 shift).

8. **AF vocabulary audit** — 15.7% hit rate is a genuine structural feature (narrative/non-directional discourse). One addition: `obra publica` as dovish. AF's signal at −0.647 z is well-supported by 413 dovish accepted hits.

9. **DLM (Battaglia & Salunina 2020) assessment** — Paper proposes LDA topic proportions → DLM state-space model → Kalman filter → latent factor. Recommendation: use as *validation* tool, not replacement. The DLM cannot replace the dictionary signal (lacks directionality). Three things to potentially take on board:
   - Fit DLM on monthly fiscal topic proportions → compare Kalman-smoothed factor with `net_tf_fwsum_z` as mutual validation
   - DLM posterior variance → propagate into BVAR as measurement error
   - Trend correction in DLM can disentangle Milei's level dominance

---

## What Still Needs Doing (Priority Order)

### Immediate (before BVAR)

1. **External validation** — Correlate `net_tf_fwsum_z` with Argentine primary balance data, EMBI+ spreads, ARS/USD exchange rate. This is the most important remaining validation step. Look for: negative correlation with primary deficit (hawkish signal → improving balance), positive correlation with EMBI+ (markets read hawkish signal as credibility). Data sources: INDEC, BCRA, JP Morgan EMBI.

2. **Git commit** — Run in terminal (git lock conflicts prevent Claude from committing):
   ```bash
   cd ~/Desktop/Masters-Project
   git add signal/dictionaries/ signal/scoring/tfidf_dictionary.py \
           data/interim/monthly_signal.csv data/interim/paragraphs_scored.csv \
           data/interim/speeches_scored.csv notebooks/02_eda_signals.ipynb
   git commit -m "Stage 4 complete: v5 scoring pipeline, negation audit, dictionary v5, monthly_signal.csv"
   ```

3. **DLM validation exercise** — Fit simple state-space on monthly fiscal topic proportions:
   - Observation: `θ_fiscal,t = intercept + l·ft + εt`
   - State: `ft = ft-1 + νt` (random walk)
   - Compare Kalman-smoothed `ft` with `net_tf_fwsum_z`
   - High correlation → strong mutual validation for BVAR

### Medium term

4. **Stage 5: LLM cross-validation** — Sample ~200 fiscal paragraphs, have Claude classify as hawkish/dovish/neutral, compare with dictionary scores. Target Spearman ρ > 0.65.

5. **BVAR construction (Stage 7)** — `econometrics/` folder is empty skeleton. Variables needed: `net_tf_fwsum_z` (fiscal signal), primary balance/GDP, inflation, output gap, EMBI+ spread. Monthly frequency, 2015M12–2026M03 (127 observations). Identification via sign restrictions or Cholesky.

---

## Running the Pipeline

```bash
cd ~/Desktop/Masters-Project
source .venv/bin/activate

# Re-run scoring (if dictionaries changed):
python signal/scoring/tfidf_dictionary.py

# Outputs written to:
#   data/interim/paragraphs_scored.csv
#   data/interim/speeches_scored.csv
#   data/interim/monthly_signal.csv
#   outputs/figures/scoring_overview.png
#   outputs/tables/scoring_summary.txt
```

---

## Key Numbers to Know

- **Corpus:** 1,498 speeches, 16,197 paragraphs, 3 presidents
- **Dictionary:** 61 hawkish + 46 dovish = 107 terms total
- **Signal:** 127 monthly observations (49 Macri + 49 AF + 29 Milei)
- **BVAR primary column:** `net_tf_fwsum_z` in `data/interim/monthly_signal.csv`
- **Negation suppression rate:** 3.4% overall (60 suppressed / 1,750 gross) — well-calibrated
- **AF low hit rate (15.7%):** known, documented, not a bug — AF's discourse is narrative-descriptive not signaling

---

## Literature Basis

- Dictionary framework: Blanchard & Leigh (2013) fiscal multipliers, Dornbusch & Edwards (1991) populist cycles, Alesina & Ardagna (2010), Kopits & Symansky (1998) fiscal rules, Barro-Gordon credibility
- DLM validation: Battaglia & Salunina (2020) LDA+DLM for fiscal policy uncertainty
- BVAR identification: Ramey (2016) government spending shocks, Blanchard-Perotti (2002)
