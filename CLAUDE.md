# Masters-Project — Claude Handoff Document
**Last updated:** 2026-05-01 (v8 keyword filter)  
**Author:** Rayyaan Kazi (BSE Master's student)  
**Project:** Fiscal hawkishness NLP signal from Argentine presidential speeches as input to a Bayesian Structural VAR

---

## Project Goal

Construct a monthly fiscal hawkishness signal from Argentine presidential speeches (Macri 2015–2019, Alberto Fernández 2019–2023, Milei 2023–2026) and use it as an identified shock series in a BVAR to estimate macroeconomic responses to fiscal stance changes.

**Primary BVAR input:** `net_hawkish_llm_z` in `data/interim/monthly_signal_llm.csv`  
**Robustness/replication check:** `net_hawkish_z` in `data/interim/monthly_signal.csv`

---

## Pipeline Overview

```
Stage 1: Scraping       signal/scraping/scraper.py
           ↓
Stage 2: Raw corpus     data/raw/speeches_raw.csv
           ↓
Stage 3: LDA            signal/topic_modeling/lda.py
           ↓  paragraphs_lda.csv (validation/narrative only — NOT load-bearing filter)
Stage 4: Dictionary     signal/scoring/tfidf_dictionary.py
           ↓  paragraphs_scored.csv, speeches_scored.csv, monthly_signal.csv
Stage 4b: Keyword audit  signal/validation/   ← NEXT (label ~150 paras, BBD audit)
           ↓  filter precision/recall report
Stage 5: LLM scoring    signal/scoring/llm_scoring.py   ← NEEDS RERUN (v8 filter)
           ↓  paragraphs_llm_scored.csv, monthly_signal_llm.csv, bvar_signal_llm.csv
Stage 5b: Few-shot      signal/scoring/llm_scoring.py --few-shot   ← PENDING
           ↓  few-shot robustness run (compare ρ with zero-shot)
Stage 5c: Validation    signal/validation/   ← PENDING
           ↓  ~60 human-labeled paragraphs, precision/recall/F1 vs LLM
Stage 6: External valid (not yet built)
           ↓  correlate net_hawkish_llm_z with primary balance, EMBI+, ARS/USD
Stage 7: BVAR           econometrics/ (not yet built)
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
| `signal/scoring/tfidf_dictionary.py` | Stage 4 scoring pipeline (v7 — keyword fiscal filter) |
| `signal/scoring/llm_scoring.py` | Stage 5 LLM paragraph scoring (Claude API) |
| `outputs/figures/llm_vs_dict_signal.png` | 4-panel LLM vs dictionary validation chart |
| `outputs/tables/scoring_summary.txt` | Dictionary scoring audit |
| `outputs/tables/llm_scoring_summary.txt` | LLM scoring audit + cross-validation stats |

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

This independence is the key validation property: the LLM and dictionary signals agree at ρ = 0.765 despite using completely different methods.

---

## Signal Results — Current State

### LLM Signal (Primary — Stage 5, zero-shot)

**Model:** claude-haiku-4-5-20251001  
**Paragraphs scored:** 2,920 fiscal keyword-filtered paragraphs (v7 filter — v8 rerun pending)  
**Aggregation:** EPU-style (H_t − D_t) / P_t, winsorised 2.5/97.5 pct, cross-president z-scored

| President | N months | Mean-z | Std-z | Score distribution (H/N/D) |
|-----------|----------|--------|-------|----------------------------|
| Macri | 48 | +0.380 | 0.648 | 48.7% / 37.6% / 13.7% |
| AF | 47 | −1.026 | 0.502 | 9.5% / 40.0% / 50.5% |
| Milei | 29 | +1.034 | 0.387 | 73.4% / 26.3% / 0.3% |

Milei–AF gap: **2.06 z-units**  
January 2024 (Davos): **fixed** — LLM correctly scores H=2, D=0 → z = +1.542 (was −1.532 in dictionary signal)

### Dictionary Signal (Robustness — Stage 4)

| President | N months | Mean-z | Std-z |
|-----------|----------|--------|-------|
| Macri | 49 | +0.247 | 0.882 |
| AF | 47 | −0.905 | 0.857 |
| Milei | 29 | +1.058 | 0.539 |

### Cross-Validation

| Comparison | Pearson r | Spearman ρ |
|------------|-----------|------------|
| LLM vs Dictionary | 0.767 | 0.765 |
| LLM vs Old signal | 0.622 | 0.648 |
| Dictionary vs Old signal | 0.608 | 0.632 |

Target ρ ≥ 0.65: **MET**. Both signals measure the same underlying construct.

**Key divergence:** January 2024 (Milei, gap = 3.07z) — LLM correctly captures Davos ideological hawkishness; dictionary missed it due to vocabulary limitations.

**Macri divergences:** LLM sees Macri as slightly more hawkish (+0.38z vs +0.25z) because it picks up fiscal-accounting hawkish content more reliably than the dictionary. Several Macri months where dictionary scored neutral, LLM scores hawkish.

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

**NOTE: v8 filter was just updated. Both tfidf_dictionary.py and llm_scoring.py must be re-run to regenerate outputs with the new fiscal paragraphs. The existing signal files are from v7.**

**Fiscal paragraph counts by president (v7 — pre-rerun):**
- Macri: 452 fiscal paragraphs / 5,247 total (8.6%) — 9.2 fiscal paras/month
- AF: 825 fiscal paragraphs / 6,264 total (13.2%) — 16.8 fiscal paras/month
- Milei: 1,643 fiscal paragraphs / 4,686 total (35.1%) — 56.7 fiscal paras/month

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

### Human validation (pending — important for thesis defense)

Every credible paper in this literature reports classification accuracy against human-labeled examples. Need to:
1. Manually label ~60 paragraphs (20 per class) — ~2 hours work
2. Compare human labels to LLM classifications
3. Report precision / recall / F1 (standard academic requirement)
4. Place labeled examples in `signal/validation/human_labels.csv`

Without this, the thesis is vulnerable to "how do you know the LLM is doing what you think?" questions.

---

## Scoring Pipelines — Technical Notes

### Stage 4: tfidf_dictionary.py (v8)

- **Keyword fiscal filter (v8):** 22 keywords — BBD (2016) Appendix B approach; replaces LDA threshold (v6) and v7 15-keyword list
- **EPU paragraph counting:** each fiscal para casts binary vote has_hawkish / has_dovish (Baker, Bloom & Davis 2016)
- **Winsorisation:** 2.5/97.5 pct before z-scoring
- **Two-tier negation:** Tier 1 strong (10-word), Tier 2 weak (3-word)
- **Direction-aware elimination:** suppresses dovish hits only
- **Negation suppression rate:** ~3.4% overall (from v7 run; will update after v8 rerun)

### Stage 5: llm_scoring.py

```bash
cd ~/Desktop/Masters-Project
source .venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."

python signal/scoring/llm_scoring.py --dry-run     # cost estimate (~$0.82)
python signal/scoring/llm_scoring.py               # full run, ~292 API calls
python signal/scoring/llm_scoring.py --model claude-sonnet-4-6  # higher accuracy
```

Checkpoint/resume: `data/interim/llm_scores_checkpoint.json` — delete to rescore from scratch.

---

## What Still Needs Doing (Priority Order)

### Immediate (v8 filter rerun)

1. **Re-run Stage 4** — `python signal/scoring/tfidf_dictionary.py` — regenerates `paragraphs_scored.csv` with v8 `is_fiscal` flags. Check new fiscal paragraph counts by president vs v7 numbers above.

2. **Re-run Stage 5** — Delete `data/interim/llm_scores_checkpoint.json`, then `python signal/scoring/llm_scoring.py --dry-run` to check cost, then full run. The checkpoint mechanism will only score newly added paragraphs if the checkpoint is kept (but safer to delete and rescore fully for consistency). Expected cost ~$0.90–$1.10 given more fiscal paragraphs from v8.

### Signal robustness (complete before BVAR)

3. **BBD keyword audit + human validation** — Label ~150 paragraphs as (a) fiscal/non-fiscal and (b) hawkish/neutral/dovish in one pass (~2–3 hours). This delivers two things simultaneously: the BBD-style filter precision/recall table (Baker, Bloom & Davis 2016 validate their keyword filter with a human audit of 12,000 articles), and the LLM classification accuracy table (precision/recall/F1 vs human labels). Both are required for a credible methodology section. Place output in `signal/validation/human_labels.csv`.

4. **Few-shot robustness run** — Add 9 labeled examples to prompt in `llm_scoring.py`. Delete checkpoint. Re-run. Compare monthly ρ between zero-shot and few-shot (target ρ > 0.85). Zero-shot is the primary result; few-shot is the robustness check (Bank of England 2025 SWP 1127 validates this approach).

5. **Ideology paragraph test** — Score ~150 ideology-only paragraphs (stratified by president) with LLM. If >20% score ±1 for Milei, consider including them. Costs ~$0.04. Note: v8 `emision` keyword will now catch some of these automatically.

6. **Sentence-level robustness** — Split fiscal paragraphs into sentences, score with LLM, compare monthly aggregations. If ρ > 0.85 vs paragraph-level, document and move on (Gentzkow, Kelly & Taddy 2019 recommend checking document unit sensitivity).

### External validation (Stage 6)

5. **Correlate signal with Argentine macro data:**
   - Primary balance/GDP (Ministry of Economy) — expect negative correlation with hawkishness
   - EMBI+ Argentina (JPMorgan) — expect negative correlation (hawkish → lower risk)
   - ARS/USD parallel rate — expect negative correlation (hawkish → peso appreciation)
   - This is required for the Bernoth-style communication shock identification

6. **Communication shock extraction** — Following Bernoth (2025) Section 5:
   - Regress `net_hawkish_llm_z` on: inflation, primary balance, EMBI+, lagged signal, president FE
   - Residual = unexpected fiscal communication shock
   - This is the BVAR input, not the raw signal
   - President fixed effects control for level differences across regimes

### BVAR (Stage 7)

7. **BVAR construction** — `econometrics/` folder is empty skeleton.
   - Variables: fiscal communication shock (residual), primary balance/GDP, inflation (INDEC), output gap, EMBI+ spread, REM inflation expectations
   - Monthly frequency, 2015M12–2026M03 (~127 observations)
   - Identification: Cholesky (shock ordered first) following Bernoth (2025)
   - Consider TVP-VAR (Primiceri 2005) as primary given structural break across presidents
   - Minnesota prior, p=2 lags (select by marginal likelihood)

### Documentation

8. **Git commit** — Run in terminal:
   ```bash
   cd ~/Desktop/Masters-Project
   git add signal/ data/interim/ data/processed/ CLAUDE.md README.md outputs/
   git commit -m "Stage 5 complete: LLM scoring, zero-shot, rho=0.765, Jan2024 fixed"
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
- **Fiscal paragraphs scored by LLM:** 2,920 (452 Macri + 825 AF + 1,643 Milei)
- **LLM primary BVAR column:** `net_hawkish_llm_z` in `data/interim/monthly_signal_llm.csv`
- **Dictionary robustness column:** `net_hawkish_z` in `data/interim/monthly_signal.csv`
- **Cross-validation:** Pearson r = 0.767, Spearman ρ = 0.765 (LLM vs dictionary)
- **January 2024 fix:** LLM z = +1.542 (was −1.532 in dictionary) — Davos speech correctly hawkish
- **President ordering (LLM):** Milei +1.03z > Macri +0.38z > AF −1.03z (gap = 2.06z)
- **Ideology-only paragraphs (excluded):** 1,593 — Milei 1,106 / AF 369 / Macri 118
- **Fiscal keyword filter:** 22 keywords v8 (symmetric — BBD 2016 approach; v8 just updated, rerun pending)
- **LLM model used:** claude-haiku-4-5-20251001, temperature=0

---

## Literature Basis

- **Fiscal keyword filter:** Baker, Bloom & Davis (2016) EPU keyword-in-text approach, Appendix B fiscal category — direct methodological basis for the 22-keyword filter and human audit requirement
- **Filter design / text methods survey:** Ash & Hansen (2023) *Text Algorithms in Economics*, Annual Review of Economics — embedding approach considered but rejected (anisotropy); BBD keyword approach retained
- **Aggregation formula:** Baker, Bloom & Davis (2016) (H_t − D_t) / P_t EPU paragraph-counting formula
- **LLM scoring:** Hansen & Kazinnik (2024) prompted LLM for central bank text (FOMC); Bank of England (2025) SWP 1127 tens-of-shot classification; IMF (2025) WP 2025/109 large-scale LLM central bank speech scoring
- **BVAR framework:** Bernoth (2025, DIW dp2137) communication shock identification and BVAR specification; Istrefi & Piloiu (2014) inflation expectations BVAR
- **Communication shock:** Bernoth (2025) residual regression on macro fundamentals; president FE for level differences
- **Dictionary basis:** Blanchard & Leigh (2013) fiscal multipliers; Dornbusch & Edwards (1991) populist cycles; Alesina & Ardagna (2010); Kopits & Symansky (1998) fiscal rules; Barro-Gordon credibility
- **Document unit sensitivity:** Gentzkow, Kelly & Taddy (2019) text-as-data survey — basis for sentence-level robustness check
- **TVP-VAR:** Primiceri (2005) time-varying structural VARs
