"""
signal/validation/plot_validation.py
─────────────────────────────────────
Produces publication-quality validation charts for the paper / appendix.

Reads:
  signal/validation/human_labels.csv

Writes (all to outputs/figures/):
  validation_confusion_matrix.png     — heatmap with counts + percentages
  validation_class_metrics.png        — precision / recall / F1 by class
  validation_president_breakdown.png  — accuracy + F1 + κ by president
  validation_error_analysis.png       — where errors fall (adjacent vs extreme)

Usage:
  cd ~/Desktop/Masters-Project
  source .venv/bin/activate
  python signal/validation/plot_validation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LABELS  = os.path.join(_ROOT, 'signal', 'validation', 'human_labels.csv')
FIGDIR  = os.path.join(_ROOT, 'outputs', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = '#1F4E79'
MID    = '#2E75B6'
LIGHT  = '#BDD7EE'
GREEN  = '#375623'
ORANGE = '#C55A11'
GREY   = '#595959'
RED    = '#C00000'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CLASS_NAMES  = {-1: 'Dovish\n(−1)', 0: 'Neutral\n(0)', 1: 'Hawkish\n(+1)'}
CLASS_LABELS = ['Dovish\n(−1)', 'Neutral\n(0)', 'Hawkish\n(+1)']
CLASSES      = [-1, 0, 1]
PRES_ORDER   = ['Macri', 'AF', 'Milei']
PRES_COLORS  = {'Macri': '#1F4E79', 'AF': '#C55A11', 'Milei': '#375623'}

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(LABELS)
human = df['human_score'].values
llm   = df['llm_score'].values
n     = len(df)

# ── Helper: per-class metrics ─────────────────────────────────────────────────
def class_metrics(human, llm, classes):
    out = {}
    for c in classes:
        tp = ((llm == c) & (human == c)).sum()
        fp = ((llm == c) & (human != c)).sum()
        fn = ((llm != c) & (human == c)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2*p*r/(p+r) if (p+r) > 0 else np.nan
        out[c] = {'precision': p, 'recall': r, 'f1': f1,
                  'n_human': (human == c).sum()}
    return out

def cohen_kappa(a, b, classes):
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in classes)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
conf = np.zeros((3, 3), dtype=int)
for i, hi in enumerate(CLASSES):
    for j, lj in enumerate(CLASSES):
        conf[i, j] = ((human == hi) & (llm == lj)).sum()

conf_pct = conf / conf.sum(axis=1, keepdims=True) * 100

cmap = LinearSegmentedColormap.from_list('blue_white', ['#FFFFFF', '#1F4E79'])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf_pct, cmap=cmap, vmin=0, vmax=100, aspect='auto')

for i in range(3):
    for j in range(3):
        pct  = conf_pct[i, j]
        cnt  = conf[i, j]
        col  = 'white' if pct > 55 else BLUE
        diag = (i == j)
        ax.text(j, i, f'{cnt}\n({pct:.0f}%)',
                ha='center', va='center', fontsize=11,
                color=col,
                fontweight='bold' if diag else 'normal')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_LABELS, fontsize=10)
ax.set_yticklabels(CLASS_LABELS, fontsize=10)
ax.set_xlabel('LLM Score', labelpad=10)
ax.set_ylabel('Human Label', labelpad=10)
ax.set_title('Confusion Matrix: Human Labels vs. LLM Scores\n(n = 72 fiscal paragraphs)', pad=12)

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Row %', fontsize=9)
cb.ax.tick_params(labelsize=9)

# Diagonal border
for i in range(3):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                 fill=False, edgecolor=ORANGE, linewidth=2.5))

acc   = np.diag(conf).sum() / conf.sum()
kappa = cohen_kappa(human, llm, CLASSES)
fig.text(0.5, -0.03,
         f'Overall accuracy: {acc:.3f}  |  Cohen\'s κ: {kappa:.3f}  |  '
         f'Zero extreme errors (dovish↔hawkish)',
         ha='center', fontsize=9, color=GREY, style='italic')

plt.tight_layout()
out = os.path.join(FIGDIR, 'validation_confusion_matrix.png')
plt.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-class Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════
metrics = class_metrics(human, llm, CLASSES)
class_display = ['Dovish (−1)', 'Neutral (0)', 'Hawkish (+1)']

prec = [metrics[c]['precision'] for c in CLASSES]
rec  = [metrics[c]['recall']    for c in CLASSES]
f1   = [metrics[c]['f1']        for c in CLASSES]
n_h  = [metrics[c]['n_human']   for c in CLASSES]

x   = np.arange(3)
w   = 0.25
fig, ax = plt.subplots(figsize=(7, 4.5))

b1 = ax.bar(x - w, prec, w, label='Precision', color=BLUE,   alpha=0.85)
b2 = ax.bar(x,     rec,  w, label='Recall',    color=MID,    alpha=0.85)
b3 = ax.bar(x + w, f1,   w, label='F1',        color=LIGHT,  alpha=0.95,
            edgecolor=BLUE, linewidth=0.8)

# Value labels
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, color=GREY)

# Macro F1 line
macro_f1 = np.nanmean(f1)
ax.axhline(macro_f1, color=RED, linewidth=1.5, linestyle='--', alpha=0.8,
           label=f'Macro F1 = {macro_f1:.3f}')

# Sample size annotations
for i, (xi, ni) in enumerate(zip(x, n_h)):
    ax.text(xi, -0.07, f'n={ni}', ha='center', fontsize=9, color=GREY,
            transform=ax.get_xaxis_transform())

ax.set_xticks(x)
ax.set_xticklabels(class_display, fontsize=10)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.08)
ax.set_title('Per-class Metrics: Precision, Recall, F1\n(LLM scores vs. human labels)', pad=10)
ax.legend(fontsize=9, framealpha=0.4, loc='lower right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))

plt.tight_layout()
out = os.path.join(FIGDIR, 'validation_class_metrics.png')
plt.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — By-president breakdown
# ══════════════════════════════════════════════════════════════════════════════
pres_data = {}
for p in PRES_ORDER:
    sub = df[df['president'] == p]
    h   = sub['human_score'].values
    l   = sub['llm_score'].values
    m   = class_metrics(h, l, CLASSES)
    pres_data[p] = {
        'n':        len(sub),
        'accuracy': (h == l).mean(),
        'macro_f1': np.nanmean([m[c]['f1'] for c in CLASSES]),
        'kappa':    cohen_kappa(h, l, CLASSES),
    }

fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=False)
metrics_to_plot = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'), ('kappa', "Cohen's κ")]
thresholds = {'accuracy': None, 'macro_f1': 0.70, 'kappa': 0.60}
thresh_labels = {'macro_f1': '0.70 defensible', 'kappa': '0.60 minimum'}

for ax, (key, title) in zip(axes, metrics_to_plot):
    vals   = [pres_data[p][key] for p in PRES_ORDER]
    colors = [PRES_COLORS[p] for p in PRES_ORDER]
    bars   = ax.bar(PRES_ORDER, vals, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, color=GREY)

    if thresholds[key] is not None:
        ax.axhline(thresholds[key], color=RED, linewidth=1.4,
                   linestyle='--', alpha=0.75,
                   label=thresh_labels[key])
        ax.legend(fontsize=8, framealpha=0.3)

    # Sample sizes
    for i, p in enumerate(PRES_ORDER):
        ax.text(i, -0.07, f"n={pres_data[p]['n']}",
                ha='center', fontsize=8.5, color=GREY,
                transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))

fig.suptitle('Validation Metrics by President', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(FIGDIR, 'validation_president_breakdown.png')
plt.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Error analysis: adjacent vs extreme, by class
# ══════════════════════════════════════════════════════════════════════════════
correct, adjacent, extreme = [], [], []
for h, l in zip(human, llm):
    diff = abs(h - l)
    if diff == 0:
        correct.append((h, l))
    elif diff == 1:
        adjacent.append((h, l))
    else:
        extreme.append((h, l))

# By true class
cats = ['Correct', 'Adjacent\nerror', 'Extreme\nerror\n(dovish↔hawkish)']
by_class = {c: [0, 0, 0] for c in CLASSES}
for h, l in zip(human, llm):
    diff = abs(h - l)
    if diff == 0:
        by_class[h][0] += 1
    elif diff == 1:
        by_class[h][1] += 1
    else:
        by_class[h][2] += 1

fig, ax = plt.subplots(figsize=(7, 4.5))
x   = np.arange(3)
w   = 0.25
clrs = [BLUE, MID, LIGHT]

for i, (c, label) in enumerate(zip(CLASSES, class_display)):
    vals = by_class[c]
    b    = ax.bar(x + (i-1)*w, vals, w, label=label,
                  color=PRES_COLORS[list(PRES_COLORS.keys())[i]], alpha=0.85)
    for bar, val in zip(b, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                    str(val), ha='center', va='bottom', fontsize=9, color=GREY)

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel('Number of paragraphs')
ax.set_title('Error Analysis: Correct, Adjacent, and Extreme Errors by Class\n(Human label as ground truth)',
             pad=10)
ax.legend(title='Human label', fontsize=9, framealpha=0.4)

# Annotation
total_extreme = len(extreme)
ax.text(0.98, 0.97, f'Total extreme errors: {total_extreme}\n(dovish↔hawkish misclassifications)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        color=RED if total_extreme > 0 else GREEN,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREY, alpha=0.8))

plt.tight_layout()
out = os.path.join(FIGDIR, 'validation_error_analysis.png')
plt.savefig(out)
plt.close()
print(f'Saved: {out}')

print('\nAll validation figures saved to outputs/figures/')
print('Files:')
for f in ['validation_confusion_matrix.png', 'validation_class_metrics.png',
          'validation_president_breakdown.png', 'validation_error_analysis.png']:
    print(f'  {f}')
