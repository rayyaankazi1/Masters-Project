"""
signal/topic_modeling/wordclouds.py────────────────────────────────────
Descriptive word cloud visualisations built on top of the LDA paragraph
dataframe produced by lda.py.

Four sets of outputs
─────────────────────
1. Whole-corpus clouds     — full tokenised vocabulary, all presidents combined
                             then per president. Shows overall rhetorical profile.
2. Fiscal-filtered clouds  — paragraphs with fiscal_topic_prob ≥ threshold,
                             all presidents combined then per president.
3. Ideology-filtered clouds— paragraphs with ideology_topic_prob ≥ threshold,
                             all presidents combined then per president.
4. Comparison panels       — fiscal vs non-fiscal side by side for each
                             president. Key validation exhibit: demonstrates
                             the filter selects meaningfully different language.

Reads
-----
  data/interim/paragraphs_lda.csv   (produced by lda.py)

Writes
------
  outputs/figures/wc_corpus_all.png
  outputs/figures/wc_corpus_<president>.png
  outputs/figures/wc_fiscal_all.png
  outputs/figures/wc_fiscal_<president>.png
  outputs/figures/wc_ideology_all.png
  outputs/figures/wc_ideology_<president>.png
  outputs/figures/wc_compare_<president>.png
"""

import os
import re
import unicodedata

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_CSV     = os.path.join(_ROOT, "data", "interim", "paragraphs_lda.csv")
FIGURES_DIR  = os.path.join(_ROOT, "outputs", "figures")

# ── Config ────────────────────────────────────────────────────────────────────
FISCAL_MIN_PROB = 0.25
PRES_ORDER      = ["Macri", "AF", "Milei"]
PRES_COLORS     = {
    "Macri": ("#BBDEFB", "#1565C0"),   # light blue, dark blue
    "AF":    ("#C8E6C9", "#1B5E20"),   # light green, dark green
    "Milei": ("#FFE0B2", "#BF360C"),   # light orange, dark orange
}
MAX_WORDS = 120
WC_WIDTH  = 1400
WC_HEIGHT = 700

# ── Phrase detection (visual only — independent of LDA) ───────────────────────
# Phrases are detected here purely for display quality in word clouds.
# A lower threshold than LDA (15 vs the LDA's 40) is fine — visual noise
# in a cloud is far less harmful than topic coherence degradation.
WC_PHRASES_MIN_COUNT = 5
WC_PHRASES_THRESHOLD = 15

# Compounds to suppress even if detected — geographic and ceremonial noise.
WC_PHRASE_BLACKLIST: set[str] = {
    "buenos_aires", "ciudad_buenos_aires", "mar_plata", "rio_negro",
    "santa_fe", "san_juan", "san_luis", "la_plata", "tierra_fuego",
    "muchas_gracias", "muy_bien", "muy_importante", "por_favor",
    "mas_alla", "dia_dia", "buenas_tardes", "buenas_noches",
    "america_latina", "estados_unidos", "naciones_unidas", "union_europea",
}

# ── Stopwords ─────────────────────────────────────────────────────────────────
# Keep in sync with lda.py — any term you don't want in clouds goes here
STOPWORDS = {
    "el","la","los","las","un","una","unos","unas",
    "a","ante","bajo","con","contra","de","desde","durante","en","entre",
    "hacia","hasta","para","por","segun","sin","sobre","tras","del","al",
    "e","ni","o","u","que","y","pero","sino","aunque","porque","si",
    "como","cuando","donde","mientras","pues","ya",
    "yo","tu","el","ella","nosotros","vosotros","ellos","ellas",
    "me","te","se","nos","os","le","les","lo",
    "este","esta","estos","estas","ese","esa","esos","esas",
    "aquel","aquella","aquellos","aquellas","ello",
    "nuestro","nuestra","nuestros","nuestras","su","sus","mi","mis",
    "es","son","era","eran","fue","fueron","ser","estar","sido","siendo",
    "ha","han","he","hemos","haber","hay","habia","hubo","habra",
    "tiene","tienen","tener","tenemos","tuvo","tenia","tenian",
    "hace","hacen","hacer","hizo","haria","haran",
    "dijo","dice","decir","dicho","decia",
    "va","van","ir","vamos","vaya","iba","iban",
    "puede","pueden","poder","pudo","podia","podran",
    "quiero","quiere","quieren","querer","queria",
    "sabe","saben","saber","supo",
    "no","mas","muy","bien","tambien","ya","solo","aun","asi",
    "aqui","ahi","alli","hoy","ayer","siempre","nunca","antes","despues",
    "todo","todos","toda","todas","cada","mucho","mucha","muchos","muchas",
    "poco","menos","otro","otra","otros","otras","mismo","misma",
    "entonces","bueno","realmente","tan","aca","alla","claro","vez","veces",
    "creo","parece","manera","parte","forma","punto","lugar","momento",
    "cosa","cosas","tipo","caso","casos",
    "eso","esto","esos","estos","esa","esas","ese",
    "uno","usted","ustedes","estan","voy","vos","van","ver","sea","mil",
    "fue","era","han","hay","les","sus","son","ser",
    "aun","mas","nos","esa","ese","eso","esos","esas",
    "ademas","tanto","cual","ahora","teniamos","ello","aquello",
    "aquel","cuyo","cuya","cuales","ambos","luego","recien",
    "vale","claro","obvio","igual","incluso","tampoco",
    "argentina","argentino","argentinos","argentinas","pais","paises",
    "nacion","republica","gobierno","presidente","presidenta",
    "senor","senora","gracias","aplausos","pueblo","hombre","mujer",
    "dia","dias","ano","anos","mes","meses","semana","semanas",
    "mundo","gente","personas","ciudadanos","sociedad",
    # extra verb forms that appear in full-corpus clouds
    "estamos","ibamos","tuvimos","hicimos","logramos","fuimos",
    "podemos","queremos","tenemos","sabemos","vamos","somos",
    "estaba","estaban","habia","habian","hubiera","hubieron",
    "haciendo","siendo","teniendo","pudiendo","dando","llegando",
    "algo","algun","alguna","algunos","algunas","nadie","nada",
    "decimos","diciendo","dijeron","dijimos","dirian",
    "tres","cuatro","cinco","seis","siete","ocho","nueve","diez",
    "dos","primer","primero","primera","segundo","segunda","ultimo","ultima",
    "gran","grande","grandes","nuevo","nueva","nuevos","nuevas",
    "verdad","hecho","tema","temas","nivel","niveles","numero","numeros",
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
        if len(w) >= 4 and w not in STOPWORDS
    ]

def build_phrase_models(para_df: pd.DataFrame) -> tuple[Phraser, Phraser]:
    """
    Fit bigram + trigram phrase models on the paragraph corpus.
    Uses a permissive threshold (WC_PHRASES_THRESHOLD=15) — fine for
    display quality, independent of the LDA's stricter settings.
    """
    texts = para_df["text_para"].astype(str).apply(tokenise).tolist()
    bigram_model  = Phrases(texts,
                            min_count=WC_PHRASES_MIN_COUNT,
                            threshold=WC_PHRASES_THRESHOLD)
    trigram_model = Phrases(bigram_model[texts],
                            min_count=WC_PHRASES_MIN_COUNT,
                            threshold=WC_PHRASES_THRESHOLD)
    return Phraser(bigram_model), Phraser(trigram_model)

def apply_phrases(tokens: list[str],
                  bigram: Phraser,
                  trigram: Phraser) -> list[str]:
    tokens = list(bigram[tokens])
    tokens = list(trigram[tokens])
    # Split any blacklisted compounds back into unigrams
    result = []
    for t in tokens:
        if t in WC_PHRASE_BLACKLIST:
            result.extend(t.split("_"))
        else:
            result.append(t)
    return result

# Module-level phrase model cache — built once in run(), used by _df_to_text
_bigram:  Phraser | None = None
_trigram: Phraser | None = None

def _df_to_text(df: pd.DataFrame) -> str:
    """Tokenise paragraphs, apply phrase models, join for WordCloud."""
    tokens_list = df["text_para"].astype(str).apply(
        lambda t: apply_phrases(tokenise(t), _bigram, _trigram)
        if _bigram is not None else tokenise(t)
    )
    return " ".join(" ".join(toks) for toks in tokens_list)

def make_wc(text: str, bg_color: str, colormap: str) -> WordCloud:
    return WordCloud(
        width=WC_WIDTH,
        height=WC_HEIGHT,
        background_color=bg_color,
        colormap=colormap,
        max_words=MAX_WORDS,
        collocations=False,
        min_word_length=4,
        stopwords=STOPWORDS,
        prefer_horizontal=0.8,
    ).generate(text)

def save_wc(wc: WordCloud, title: str, path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Shared helper ─────────────────────────────────────────────────────────────

def _df_to_text(df: pd.DataFrame) -> str:
    """Tokenise paragraphs and join into a single string for WordCloud."""
    tokens_list = df["text_para"].astype(str).apply(tokenise)
    return " ".join(" ".join(toks) for toks in tokens_list)


# ── 1. Whole-corpus clouds ────────────────────────────────────────────────────

def plot_full_corpus(para_df: pd.DataFrame):
    print("\n── Whole-corpus word clouds ──────────────────────────────────────")

    # All presidents combined
    text_all = _df_to_text(para_df)
    wc = make_wc(text_all, bg_color="white", colormap="plasma")
    save_wc(wc, "Full corpus vocabulary — all presidents",
            os.path.join(FIGURES_DIR, "wc_corpus_all.png"))

    # Per president
    for pres in PRES_ORDER:
        sub = para_df[para_df["president"] == pres]
        if sub.empty:
            continue
        text = _df_to_text(sub)
        if len(text.strip()) < 50:
            print(f"  {pres}: not enough text — skipping.")
            continue
        wc = make_wc(text, bg_color="white", colormap=(
            "Blues" if pres == "Macri" else
            "Greens" if pres == "AF" else
            "Oranges"
        ))
        save_wc(wc, f"Full corpus vocabulary — {pres}",
                os.path.join(FIGURES_DIR, f"wc_corpus_{pres.lower()}.png"))

# ── 2. Fiscal-filtered clouds ─────────────────────────────────────────────────

def plot_fiscal_filtered(para_df: pd.DataFrame):
    print("\n── Fiscal-filtered word clouds ───────────────────────────────────")
    if "fiscal_topic_prob" not in para_df.columns:
        print("  fiscal_topic_prob not found — run lda.py first. Skipping.")
        return
    fiscal = para_df[para_df["fiscal_topic_prob"] >= FISCAL_MIN_PROB]
    print(f"  {len(fiscal):,} paragraphs pass fiscal threshold (≥{FISCAL_MIN_PROB})")

    # All presidents combined
    text_all = _df_to_text(fiscal)
    wc = make_wc(text_all, bg_color="white", colormap="YlOrRd")
    save_wc(wc, f"Fiscal vocabulary — all presidents (prob ≥ {FISCAL_MIN_PROB})",
            os.path.join(FIGURES_DIR, "wc_fiscal_all.png"))

    # Per president
    for pres in PRES_ORDER:
        sub = fiscal[fiscal["president"] == pres]
        if sub.empty:
            print(f"  {pres}: no fiscal paragraphs — skipping.")
            continue
        text = _df_to_text(sub)
        wc = make_wc(text, bg_color="white", colormap=(
            "Blues" if pres == "Macri" else
            "Greens" if pres == "AF" else
            "Oranges"
        ))
        save_wc(
            wc,
            f"Fiscal vocabulary — {pres} (prob ≥ {FISCAL_MIN_PROB})",
            os.path.join(FIGURES_DIR, f"wc_fiscal_{pres.lower()}.png"),
        )

# ── 3. Ideology-filtered clouds ───────────────────────────────────────────────

def plot_ideology_filtered(para_df: pd.DataFrame):
    print("\n── Ideology-filtered word clouds ─────────────────────────────────")
    if "ideology_topic_prob" not in para_df.columns:
        print("  ideology_topic_prob not found — run lda.py with "
              "IDEOLOGY_TOPIC_IDS set. Skipping.")
        return
    if (para_df["ideology_topic_prob"] == 0).all():
        print("  ideology_topic_prob is all 0.0 — IDEOLOGY_TOPIC_IDS was empty "
              "when lda.py last ran. Skipping.")
        return
    ideo = para_df[para_df["ideology_topic_prob"] >= FISCAL_MIN_PROB]
    print(f"  {len(ideo):,} paragraphs pass ideology threshold (≥{FISCAL_MIN_PROB})")

    # All presidents combined
    text_all = _df_to_text(ideo)
    wc = make_wc(text_all, bg_color="white", colormap="PuRd")
    save_wc(wc, f"Ideology vocabulary — all presidents (prob ≥ {FISCAL_MIN_PROB})",
            os.path.join(FIGURES_DIR, "wc_ideology_all.png"))

    # Per president
    for pres in PRES_ORDER:
        sub = ideo[ideo["president"] == pres]
        if sub.empty:
            print(f"  {pres}: no ideology paragraphs — skipping.")
            continue
        text = _df_to_text(sub)
        wc = make_wc(text, bg_color="white", colormap=(
            "Blues" if pres == "Macri" else
            "Greens" if pres == "AF" else
            "Oranges"
        ))
        save_wc(
            wc,
            f"Ideology vocabulary — {pres} (prob ≥ {FISCAL_MIN_PROB})",
            os.path.join(FIGURES_DIR, f"wc_ideology_{pres.lower()}.png"),
        )


# ── 4. Side-by-side comparison panels ────────────────────────────────────────

def plot_comparison(para_df: pd.DataFrame):
    """
    For each president: fiscal paragraphs (left) vs non-fiscal paragraphs (right).
    This is the key validation exhibit — shows the filter selects meaningfully
    different language, not just a random subset of the full corpus.
    """
    print("\n── Comparison panels (fiscal vs non-fiscal) ──────────────────────")
    for pres in PRES_ORDER:
        sub      = para_df[para_df["president"] == pres]
        fiscal    = sub[sub["fiscal_topic_prob"] >= FISCAL_MIN_PROB]
        nonfiscal = sub[sub["fiscal_topic_prob"] <  FISCAL_MIN_PROB]

        if fiscal.empty or nonfiscal.empty:
            print(f"  {pres}: insufficient data for comparison — skipping.")
            continue

        def get_text(df):
            return " ".join(df["text_para"].astype(str).apply(
                lambda t: " ".join(tokenise(t))
            ))

        text_f  = get_text(fiscal)
        text_nf = get_text(nonfiscal)

        cmap = "Blues" if pres == "Macri" else "Greens" if pres == "AF" else "Oranges"

        wc_f  = make_wc(text_f,  bg_color="white", colormap=cmap)
        wc_nf = make_wc(text_nf, bg_color="#F5F5F5", colormap="Greys")

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(
            f"Fiscal vs non-fiscal vocabulary — {pres}  "
            f"({len(fiscal):,} fiscal / {len(nonfiscal):,} non-fiscal paragraphs)",
            fontsize=13, fontweight="bold"
        )

        axes[0].imshow(wc_f, interpolation="bilinear")
        axes[0].axis("off")
        axes[0].set_title(
            f"Fiscal paragraphs (prob ≥ {FISCAL_MIN_PROB})",
            fontsize=11, color="#333333"
        )

        axes[1].imshow(wc_nf, interpolation="bilinear")
        axes[1].axis("off")
        axes[1].set_title(
            "Non-fiscal paragraphs",
            fontsize=11, color="#555555"
        )

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"wc_compare_{pres.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    global _bigram, _trigram
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loading paragraph dataframe from {PARA_CSV}...")
    if not os.path.exists(PARA_CSV):
        raise FileNotFoundError(
            f"{PARA_CSV} not found. Run signal/topic_modeling/lda.py first."
        )

    para_df = pd.read_csv(PARA_CSV)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)]

    print(f"  {len(para_df):,} paragraphs loaded")
    for col, label in [("fiscal_topic_prob", "fiscal"), ("ideology_topic_prob", "ideology")]:
        if col in para_df.columns:
            n = (para_df[col] >= FISCAL_MIN_PROB).sum()
            pct = n / len(para_df) * 100
            print(f"  {label.capitalize()} paragraphs (≥{FISCAL_MIN_PROB}): {n:,} ({pct:.1f}%)")

    print(f"\nBuilding phrase models for word clouds "
          f"(min_count={WC_PHRASES_MIN_COUNT}, threshold={WC_PHRASES_THRESHOLD})...")
    _bigram, _trigram = build_phrase_models(para_df)
    print("  Done.")

    plot_full_corpus(para_df)
    plot_fiscal_filtered(para_df)
    plot_ideology_filtered(para_df)
    plot_comparison(para_df)

    total = len([
        f for f in os.listdir(FIGURES_DIR)
        if f.startswith("wc_") and f.endswith(".png")
    ])
    print(f"\nDone — {total} word cloud files in {FIGURES_DIR}/")


if __name__ == "__main__":
    run()
