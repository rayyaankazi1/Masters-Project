"""
signal/topic_modeling/lda.py
─────────────────────────────
Stage 3 of the signal pipeline (README §Pipeline 1 — Stage 3).

Two roles (per README):
  1. Fiscal-relevance filtering — assigns each paragraph a posterior
     probability of belonging to the fiscal/monetary topic. Downstream
     scoring stages weight or filter by this probability.
  2. Descriptive validation — topic-word distributions, fiscal-topic
     share by president, and fiscal-filtered word clouds are reported
     as model-free validation exhibits.

Reads
-----
  data/raw/speeches_raw.csv

Writes
------
  data/interim/paragraphs_lda.csv        paragraph-level dataframe with
                                         topic probabilities + fiscal flag
  data/interim/lda/                      saved gensim model, dictionary,
                                         and corpus (for reproducibility)
  outputs/figures/lda_coherence.png      coherence vs k grid-search plot
  outputs/figures/lda_topics.png         top-10 words per topic (best k)
  outputs/figures/lda_fiscal_share.png   fiscal-topic share by president
  outputs/figures/wordcloud_<pres>.png   fiscal-filtered word clouds
  outputs/lda_validation.html            pyLDAvis interactive browser vis

Config
------
  Adjust NUM_TOPICS_GRID, FISCAL_TOPIC_ID, and FISCAL_MIN_PROB below.
  On first run, leave FISCAL_TOPIC_ID = None — the script auto-detects
  the fiscal topic via seed-word overlap and prints its finding. Review
  the coherence plot and topic-word chart, then hard-code the id if needed.
"""

import os
import re
import pickle
import unicodedata
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves figures without blocking
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.abspath(os.path.join(_HERE, "..", ".."))
RAW_CSV       = os.path.join(_ROOT, "data", "raw", "speeches_raw.csv")
INTERIM_DIR   = os.path.join(_ROOT, "data", "interim")
LDA_DIR       = os.path.join(INTERIM_DIR, "lda")
FIGURES_DIR   = os.path.join(_ROOT, "outputs", "figures")
OUTPUTS_DIR   = os.path.join(_ROOT, "outputs")

# ── Config ────────────────────────────────────────────────────────────────────
NUM_TOPICS_GRID = [5, 8, 10, 12, 15, 20]   # k values to search
COHERENCE_METRIC = "c_v"                    # c_v is standard for topic quality
PASSES          = 20                        # LDA training passes
RANDOM_STATE    = 42
MIN_PARA_WORDS  = 20                        # discard very short paragraphs
NO_BELOW        = 5                         # min document frequency
NO_ABOVE        = 0.55                      # max document frequency (fraction)

# After first run, inspect outputs and set this to the topic index
# that corresponds to the fiscal/monetary policy topic.
# Leave as None to use automatic seed-word detection.
FISCAL_TOPIC_ID: Optional[int] = None

# Threshold for labelling a paragraph as "fiscal-relevant"
FISCAL_MIN_PROB = 0.25

PRES_ORDER  = ["Macri", "AF", "Milei"]
PRES_COLORS = {"Macri": "#2196F3", "AF": "#4CAF50", "Milei": "#FF5722"}

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS_ES = {
    # Articles & prepositions
    "el","la","los","las","un","una","unos","unas",
    "a","ante","bajo","con","contra","de","desde","durante","en","entre",
    "hacia","hasta","para","por","segun","sin","sobre","tras","del","al",
    # Conjunctions
    "e","ni","o","u","que","y","pero","sino","aunque","porque","si",
    "como","cuando","donde","mientras","pues","ya",
    # Pronouns
    "yo","tu","el","ella","nosotros","vosotros","ellos","ellas",
    "me","te","se","nos","os","le","les","lo","les",
    "este","esta","estos","estas","ese","esa","esos","esas",
    "aquel","aquella","aquellos","aquellas","ello",
    "nuestro","nuestra","nuestros","nuestras","su","sus","mi","mis",
    # High-frequency verbs
    "es","son","era","eran","fue","fueron","ser","estar","sido","siendo",
    "ha","han","he","hemos","haber","hay","habia","hubo","habra",
    "tiene","tienen","tener","tenemos","tuvo","tenia","tenian",
    "hace","hacen","hacer","hizo","haria","haran",
    "dijo","dice","decir","dicho","decia",
    "va","van","ir","vamos","vaya","iba","iban",
    "puede","pueden","poder","pudo","podia","podran",
    "quiero","quiere","quieren","querer","queria",
    "sabe","saben","saber","supo",
    # Generic filler
    "no","mas","muy","bien","tambien","ya","solo","aun","asi",
    "aqui","ahi","alli","hoy","ayer","siempre","nunca","antes","despues",
    "todo","todos","toda","todas","cada","mucho","mucha","muchos","muchas",
    "poco","menos","otro","otra","otros","otras","mismo","misma",
    "entonces","bueno","realmente","tan","aca","alla","claro","vez","veces",
    "creo","parece","manera","parte","forma","punto","lugar","momento",
    "cosa","cosas","tipo","caso","casos",
    # Argentina-specific noise
    "argentina","argentino","argentinos","argentinas","pais","paises",
    "nacion","republica","gobierno","presidente","presidenta",
    "senor","senora","gracias","aplausos","pueblo","hombre","mujer",
    "dia","dias","ano","anos","mes","meses","semana","semanas",
    "mundo","gente","personas","ciudadanos","sociedad",
}

# Seed words for auto-detecting the fiscal topic
FISCAL_SEEDS = {
    "fiscal","deficit","superavit","gasto","presupuesto","ajuste",
    "deuda","emision","monetaria","inflacion","economia","economico",
    "crecimiento","pbi","reservas","dolar","peso","banco","financiero",
    "impuesto","recaudacion","subsidios","privatizacion","austeridad",
    "equilibrio","saneamiento","motosierra","licuadora","casta",
    "inversion","exportaciones","importaciones","produccion","empleo",
    "salario","pobreza","distribucion","redistribucion","obra",
}

# ── Text helpers ──────────────────────────────────────────────────────────────
_CLEAN_RE = re.compile(r"[^a-z\s]")

def normalise(text: str) -> str:
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return _CLEAN_RE.sub(" ", text)

def tokenise(text: str) -> list[str]:
    return [
        w for w in normalise(text).split()
        if len(w) >= 3 and w not in STOPWORDS_ES
    ]

# ── Paragraph extraction ──────────────────────────────────────────────────────

def extract_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each speech into paragraph-level rows.
    Paragraphs are newline-separated chunks of MIN_PARA_WORDS or more words.
    Returns a dataframe with one row per paragraph.
    """
    records = []
    for _, row in df.iterrows():
        lines = str(row["text_raw"]).split("\n")
        for para_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            tokens = tokenise(line)
            if len(tokens) < MIN_PARA_WORDS:
                continue
            records.append({
                "speech_id":    row["speech_id"],
                "para_idx":     para_idx,
                "date":         row["date"],
                "president":    row["president"],
                "president_id": row["president_id"],
                "year_month":   row["year_month"],
                "n_tokens":     len(tokens),
                "text_para":    line,
                "tokens":       tokens,
            })
    para_df = pd.DataFrame(records)
    para_df["para_id"] = range(len(para_df))
    return para_df

# ── Coherence grid search ─────────────────────────────────────────────────────

def grid_search_k(
    texts: list[list[str]],
    dictionary: corpora.Dictionary,
    corpus: list,
) -> tuple[int, list[float]]:
    """
    Train LDA for each k in NUM_TOPICS_GRID and return the best k
    and the list of coherence scores.
    """
    scores = []
    print(f"\nCoherence grid search over k = {NUM_TOPICS_GRID}")
    print("-" * 50)
    for k in NUM_TOPICS_GRID:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=PASSES,
            alpha="auto",
            eta="auto",
            random_state=RANDOM_STATE,
        )
        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence=COHERENCE_METRIC,
        )
        score = cm.get_coherence()
        scores.append(score)
        print(f"  k={k:2d}  coherence ({COHERENCE_METRIC}) = {score:.4f}")

    best_k = NUM_TOPICS_GRID[int(np.argmax(scores))]
    print(f"\n  Best k = {best_k} (coherence = {max(scores):.4f})")
    return best_k, scores

# ── Fiscal topic detection ────────────────────────────────────────────────────

def detect_fiscal_topic(model: LdaModel, n_words: int = 30) -> int:
    """
    Returns the topic index with the highest overlap between its top-n words
    and the FISCAL_SEEDS vocabulary. Used when FISCAL_TOPIC_ID is None.
    """
    best_topic, best_overlap = 0, 0
    for i in range(model.num_topics):
        top_words = {w for w, _ in model.show_topic(i, topn=n_words)}
        overlap = len(top_words & FISCAL_SEEDS)
        if overlap > best_overlap:
            best_overlap, best_topic = overlap, i
    return best_topic

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_coherence(scores: list[float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(NUM_TOPICS_GRID, scores, marker="o", color="steelblue", linewidth=2)
    best_k = NUM_TOPICS_GRID[int(np.argmax(scores))]
    ax.axvline(best_k, color="red", linestyle="--", linewidth=1.2,
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel(f"Coherence ({COHERENCE_METRIC})")
    ax.set_title("LDA coherence grid search")
    ax.set_xticks(NUM_TOPICS_GRID)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "lda_coherence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_topic_words(model: LdaModel, fiscal_topic_id: int, n_words: int = 10):
    k = model.num_topics
    cols = min(k, 4)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(k):
        words, weights = zip(*model.show_topic(i, topn=n_words))
        color = "#FF5722" if i == fiscal_topic_id else "steelblue"
        axes[i].barh(list(words)[::-1], list(weights)[::-1], color=color)
        label = " ★ FISCAL" if i == fiscal_topic_id else ""
        axes[i].set_title(f"Topic {i}{label}", fontsize=9)
        axes[i].tick_params(axis="y", labelsize=7)
        axes[i].tick_params(axis="x", labelsize=7)

    for j in range(k, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Top {n_words} words per topic (k={k}, fiscal=Topic {fiscal_topic_id})",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "lda_topics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_fiscal_share(para_df: pd.DataFrame):
    share = (
        para_df[para_df["president"].isin(PRES_ORDER)]
        .groupby(["president", "year_month"])
        ["is_fiscal"].mean()
        .reset_index()
    )
    share["ym_dt"] = pd.to_datetime(share["year_month"])

    fig, ax = plt.subplots(figsize=(16, 3))
    for pres in PRES_ORDER:
        sub = share[share["president"] == pres].sort_values("ym_dt")
        ax.plot(sub["ym_dt"], sub["is_fiscal"],
                label=pres, color=PRES_COLORS[pres], linewidth=1.2)

    ax.set_title(f"Fiscal-topic paragraph share by month (threshold={FISCAL_MIN_PROB})")
    ax.set_ylabel("Fraction of paragraphs")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "lda_fiscal_share.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_wordclouds(para_df: pd.DataFrame):
    """Word clouds built from fiscal-filtered paragraphs, one per president."""
    fiscal_paras = para_df[para_df["is_fiscal"]]

    for pres in PRES_ORDER:
        sub = fiscal_paras[fiscal_paras["president"] == pres]
        if sub.empty:
            print(f"  No fiscal paragraphs for {pres} — skipping word cloud.")
            continue

        text = " ".join(" ".join(t) for t in sub["tokens"])
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            colormap="Blues" if pres == "Macri" else "Greens" if pres == "AF" else "Oranges",
            max_words=100,
            collocations=False,
            min_word_length=3,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Fiscal vocabulary — {pres} (paragraphs with fiscal-topic prob ≥ {FISCAL_MIN_PROB})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"wordcloud_{pres.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run(fiscal_topic_id: Optional[int] = FISCAL_TOPIC_ID):
    for d in [INTERIM_DIR, LDA_DIR, FIGURES_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load and extract paragraphs ────────────────────────────────────────
    print("Loading speeches...")
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    df = df[df["president"].isin(PRES_ORDER)]
    print(f"  {len(df)} speeches from Macri / AF / Milei")

    print("\nExtracting paragraphs...")
    para_df = extract_paragraphs(df)
    print(f"  {len(para_df):,} paragraphs (≥{MIN_PARA_WORDS} tokens)")

    texts = para_df["tokens"].tolist()

    # ── 2. Build dictionary and corpus ────────────────────────────────────────
    print("\nBuilding vocabulary...")
    dictionary = corpora.Dictionary(texts)
    print(f"  Vocabulary before filtering: {len(dictionary):,} terms")
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    print(f"  Vocabulary after filtering : {len(dictionary):,} terms")
    corpus = [dictionary.doc2bow(t) for t in texts]

    # ── 3. Grid search over k ─────────────────────────────────────────────────
    best_k, coherence_scores = grid_search_k(texts, dictionary, corpus)
    plot_coherence(coherence_scores)

    # ── 4. Train final model with best k and auto α/β ─────────────────────────
    print(f"\nTraining final model (k={best_k}, α=auto, η=auto)...")
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        passes=PASSES,
        alpha="auto",
        eta="auto",
        random_state=RANDOM_STATE,
    )

    # ── 5. Identify fiscal topic ──────────────────────────────────────────────
    if fiscal_topic_id is None:
        fiscal_topic_id = detect_fiscal_topic(lda)
        print(f"\nAuto-detected fiscal topic: Topic {fiscal_topic_id}")
        print("  Top words:", [w for w, _ in lda.show_topic(fiscal_topic_id, topn=15)])
        print("\n  Review lda_topics.png and set FISCAL_TOPIC_ID manually if needed.")
    else:
        print(f"\nUsing pre-set fiscal topic: Topic {fiscal_topic_id}")

    plot_topic_words(lda, fiscal_topic_id)

    # ── 6. Compute per-paragraph topic probabilities ──────────────────────────
    print("\nComputing per-paragraph topic probabilities...")
    topic_probs = []
    for bow in corpus:
        doc_topics = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        fiscal_prob = doc_topics.get(fiscal_topic_id, 0.0)
        topic_probs.append({
            **{f"topic_{i}": doc_topics.get(i, 0.0) for i in range(best_k)},
            "fiscal_topic_prob": fiscal_prob,
        })

    prob_df = pd.DataFrame(topic_probs)
    para_df = pd.concat([para_df.reset_index(drop=True), prob_df], axis=1)
    para_df["is_fiscal"] = para_df["fiscal_topic_prob"] >= FISCAL_MIN_PROB
    para_df.drop(columns=["tokens"], inplace=True)

    print(f"  Fiscal paragraphs (prob ≥ {FISCAL_MIN_PROB}): "
          f"{para_df['is_fiscal'].sum():,} / {len(para_df):,} "
          f"({para_df['is_fiscal'].mean()*100:.1f}%)")

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    out_csv = os.path.join(INTERIM_DIR, "paragraphs_lda.csv")
    para_df.to_csv(out_csv, index=False)
    print(f"\nSaved paragraph dataframe: {out_csv}")

    lda.save(os.path.join(LDA_DIR, "lda_model"))
    dictionary.save(os.path.join(LDA_DIR, "dictionary"))
    with open(os.path.join(LDA_DIR, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    print(f"Saved model artefacts: {LDA_DIR}/")

    # ── 8. pyLDAvis ──────────────────────────────────────────────────────────
    print("\nGenerating pyLDAvis...")
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
    vis_path = os.path.join(OUTPUTS_DIR, "lda_validation.html")
    pyLDAvis.save_html(vis, vis_path)
    print(f"Saved interactive vis: {vis_path}")

    # ── 9. Descriptive plots ──────────────────────────────────────────────────
    print("\nGenerating descriptive plots...")
    plot_fiscal_share(para_df)
    print("\nGenerating word clouds...")

    # Re-attach tokens for word cloud generation
    para_tokens = extract_paragraphs(df)["tokens"].tolist()
    para_df["tokens"] = para_tokens
    plot_wordclouds(para_df)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print("\n=== LDA SUMMARY ===")
    print(f"Best k            : {best_k}")
    print(f"Fiscal topic id   : {fiscal_topic_id}")
    print(f"Fiscal top words  : {[w for w, _ in lda.show_topic(fiscal_topic_id, topn=10)]}")
    print(f"Fiscal paragraphs : {para_df['is_fiscal'].sum():,}")
    print("\nFiscal paragraph share by president:")
    print(
        para_df[para_df["president"].isin(PRES_ORDER)]
        .groupby("president")["is_fiscal"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "fiscal", "count": "total", "mean": "share"})
        .to_string()
    )

    return lda, para_df


if __name__ == "__main__":
    lda, para_df = run()
