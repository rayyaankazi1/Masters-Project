"""
signal/topic_modeling/wordclouds.py────────────────────────────────────
Fiscal word cloud visualisations — per-president, v8 BBD fiscal filter.

Fiscal filter: uses is_fiscal flag from paragraphs_scored.csv (BBD 2016
22-keyword approach) — NOT the old LDA fiscal_topic_prob threshold.

Reads
-----
  data/interim/paragraphs_scored.csv   (produced by tfidf_dictionary.py, v8)

Writes
------
  outputs/figures/wc_fiscal_macri.png
  outputs/figures/wc_fiscal_af.png
  outputs/figures/wc_fiscal_milei.png
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
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, "..", ".."))
PARA_CSV     = os.path.join(_ROOT, "data", "interim", "paragraphs_scored.csv")
FIGURES_DIR  = os.path.join(_ROOT, "outputs", "figures")

# ── Config ────────────────────────────────────────────────────────────────────
PRES_ORDER  = ["Macri", "AF", "Milei"]
PRES_COLORS = {
    "Macri": ("#BBDEFB", "#1565C0"),   # light blue, dark blue
    "AF":    ("#C8E6C9", "#1B5E20"),   # light green, dark green
    "Milei": ("#FFE0B2", "#BF360C"),   # light orange, dark orange
}
MAX_WORDS = 120
WC_WIDTH  = 1400
WC_HEIGHT = 700

# ── Stopwords ─────────────────────────────────────────────────────────────────
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

def _df_to_text(df: pd.DataFrame) -> str:
    """Tokenise paragraphs and join into a single string for WordCloud."""
    tokens_list = df["text_para"].astype(str).apply(tokenise)
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

# ── Fiscal-filtered clouds — per president ────────────────────────────────────

def plot_fiscal_filtered(para_df: pd.DataFrame):
    print("\n── Fiscal-filtered word clouds (v8 BBD is_fiscal flag) ───────────")
    if "is_fiscal" not in para_df.columns:
        print("  is_fiscal not found — run tfidf_dictionary.py first. Skipping.")
        return
    fiscal = para_df[para_df["is_fiscal"] == True]
    print(f"  {len(fiscal):,} paragraphs pass v8 BBD fiscal filter (is_fiscal=True)")
    for pres in PRES_ORDER:
        n = len(fiscal[fiscal["president"] == pres])
        print(f"    {pres}: {n:,} fiscal paragraphs")

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
            f"Fiscal vocabulary — {pres} (v8 BBD filter)",
            os.path.join(FIGURES_DIR, f"wc_fiscal_{pres.lower()}.png"),
        )

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loading paragraph dataframe from {PARA_CSV}...")
    if not os.path.exists(PARA_CSV):
        raise FileNotFoundError(
            f"{PARA_CSV} not found. Run signal/scoring/tfidf_dictionary.py first."
        )

    para_df = pd.read_csv(PARA_CSV)
    para_df = para_df[para_df["president"].isin(PRES_ORDER)]

    print(f"  {len(para_df):,} paragraphs loaded")
    if "is_fiscal" in para_df.columns:
        n = para_df["is_fiscal"].sum()
        pct = n / len(para_df) * 100
        print(f"  Fiscal paragraphs (v8 BBD is_fiscal=True): {n:,} ({pct:.1f}%)")

    plot_fiscal_filtered(para_df)

    total = len([
        f for f in os.listdir(FIGURES_DIR)
        if f.startswith("wc_fiscal_") and f.endswith(".png")
    ])
    print(f"\nDone — {total} fiscal word cloud files in {FIGURES_DIR}/")


if __name__ == "__main__":
    run()
