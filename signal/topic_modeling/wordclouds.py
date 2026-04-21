"""
signal/topic_modeling/wordclouds.py
────────────────────────────────────
Descriptive word cloud visualisations built on top of the LDA paragraph
dataframe produced by lda.py.

Three sets of outputs
─────────────────────
1. Whole-corpus clouds     — full tokenised vocabulary by president, no
                             topic filter. Shows overall rhetorical profile.
2. Fiscal-filtered clouds  — only paragraphs with fiscal-topic prob ≥ threshold.
                             Shows fiscal-specific vocabulary (already in lda.py
                             but reproduced here for standalone use).
3. Comparison panels       — fiscal vs non-fiscal side by side for each
                             president. This is the key validation exhibit:
                             demonstrates the fiscal filter selects meaningfully
                             different language, not a random subset.

Reads
-----
  data/interim/paragraphs_lda.csv   (produced by lda.py)
  data/raw/speeches_raw.csv         (for tokenising whole-corpus clouds)

Writes
------
  outputs/figures/wc_full_<president>.png
  outputs/figures/wc_fiscal_<president>.png
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_CSV    = os.path.join(_ROOT, "data", "interim", "paragraphs_lda.csv")
FIGURES_DIR = os.path.join(_ROOT, "outputs", "figures")

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

# ── 1. Whole-corpus clouds ────────────────────────────────────────────────────

def plot_full_corpus(para_df: pd.DataFrame):
    print("\n── Whole-corpus word clouds ──────────────────────────────────────")
    for pres in PRES_ORDER:
        sub = para_df[para_df["president"] == pres]
        if sub.empty:
            continue
        text = " ".join(sub["text_para"].astype(str).apply(
            lambda t: " ".join(tokenise(t))
        ))
        if len(text.strip()) < 50:
            print(f"  {pres}: not enough text — skipping.")
            continue
        bg, _ = PRES_COLORS[pres]
        wc = make_wc(text, bg_color="white", colormap=(
            "Blues" if pres == "Macri" else
            "Greens" if pres == "AF" else
            "Oranges"
        ))
        path = os.path.join(FIGURES_DIR, f"wc_full_{pres.lower()}.png")
        save_wc(wc, f"Full corpus vocabulary — {pres}", path)

# ── 2. Fiscal-filtered clouds ─────────────────────────────────────────────────

def plot_fiscal_filtered(para_df: pd.DataFrame):
    print("\n── Fiscal-filtered word clouds ───────────────────────────────────")
    fiscal = para_df[para_df["is_fiscal"]]
    for pres in PRES_ORDER:
        sub = fiscal[fiscal["president"] == pres]
        if sub.empty:
            print(f"  {pres}: no fiscal paragraphs — skipping.")
            continue
        text = " ".join(sub["text_para"].astype(str).apply(
            lambda t: " ".join(tokenise(t))
        ))
        wc = make_wc(text, bg_color="white", colormap=(
            "Blues" if pres == "Macri" else
            "Greens" if pres == "AF" else
            "Oranges"
        ))
        path = os.path.join(FIGURES_DIR, f"wc_fiscal_{pres.lower()}.png")
        save_wc(
            wc,
            f"Fiscal vocabulary — {pres} "
            f"(paragraphs with fiscal-topic prob ≥ {FISCAL_MIN_PROB})",
            path,
        )

# ── 3. Side-by-side comparison panels ────────────────────────────────────────

def plot_comparison(para_df: pd.DataFrame):
    """
    For each president: fiscal paragraphs (left) vs non-fiscal paragraphs (right).
    This is the key validation exhibit — shows the filter selects meaningfully
    different language, not just a random subset of the full corpus.
    """
    print("\n── Comparison panels (fiscal vs non-fiscal) ──────────────────────")
    for pres in PRES_ORDER:
        sub      = para_df[para_df["president"] == pres]
        fiscal   = sub[sub["is_fiscal"]]
        nonfiscal = sub[~sub["is_fiscal"]]

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
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loading paragraph dataframe from {PARA_CSV}...")
    if not os.path.exists(PARA_CSV):
        raise FileNotFoundError(
            f"{PARA_CSV} not found. Run signal/topic_modeling/lda.py first."
        )

    para_df = pd.read_csv(PARA_CSV)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)]

    print(f"  {len(para_df):,} paragraphs loaded")
    print(f"  Fiscal paragraphs: {para_df['is_fiscal'].sum():,} "
          f"({para_df['is_fiscal'].mean()*100:.1f}%)")
    print(f"\n  Breakdown by president:")
    print(
        para_df.groupby("president")["is_fiscal"]
        .agg(fiscal="sum", total="count")
        .loc[PRES_ORDER]
        .to_string()
    )

    plot_full_corpus(para_df)
    plot_fiscal_filtered(para_df)
    plot_comparison(para_df)

    print("\nAll word clouds saved to outputs/figures/")


if __name__ == "__main__":
    run()
