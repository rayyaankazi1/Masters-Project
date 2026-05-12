"""
signal/validation/evaluate_labels.py
──────────────────────────────────────
Compares your human labels against LLM scores and produces the validation
table needed for the thesis and workshop paper.

Reads:
  signal/validation/paragraphs_to_label.xlsx   (your completed labels)
  signal/validation/labeling_key.csv            (the private key)

Writes:
  signal/validation/human_labels.csv            (merged, archival)
  outputs/tables/human_validation_report.txt    (the table for the paper)

Usage:
  cd ~/Desktop/Masters-Project
  python signal/validation/evaluate_labels.py
"""

import os
import numpy as np
import pandas as pd

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
XLSX     = os.path.join(_ROOT, 'signal', 'validation', 'paragraphs_to_label.xlsx')
KEY      = os.path.join(_ROOT, 'signal', 'validation', 'labeling_key.csv')
OUT_CSV  = os.path.join(_ROOT, 'signal', 'validation', 'human_labels.csv')
OUT_TXT  = os.path.join(_ROOT, 'outputs', 'tables', 'human_validation_report.txt')

# ── Load labels ───────────────────────────────────────────────────────────────
try:
    labels = pd.read_excel(XLSX, sheet_name='Paragraphs', header=0)
except Exception as e:
    print(f"Could not read xlsx: {e}")
    raise

# Normalise column names (in case of whitespace)
labels.columns = labels.columns.str.strip()

# Find the score column
score_col = [c for c in labels.columns if 'score' in c.lower() or 'your' in c.lower()]
if not score_col:
    raise ValueError(f"Cannot find score column. Columns: {labels.columns.tolist()}")
score_col = score_col[0]

labels = labels[['#', score_col]].rename(columns={'#': 'item_no', score_col: 'human_score'})
labels = labels.dropna(subset=['human_score'])
labels['item_no'] = labels['item_no'].astype(int)
labels['human_score'] = pd.to_numeric(labels['human_score'], errors='coerce')
labels = labels.dropna(subset=['human_score'])
labels['human_score'] = labels['human_score'].astype(int)

valid_scores = {-1, 0, 1}
invalid = labels[~labels['human_score'].isin(valid_scores)]
if len(invalid) > 0:
    print(f"WARNING: {len(invalid)} rows have invalid scores (not -1/0/1) — dropping:")
    print(invalid)
    labels = labels[labels['human_score'].isin(valid_scores)]

print(f"Loaded {len(labels)} labeled paragraphs")

# ── Load key ──────────────────────────────────────────────────────────────────
key = pd.read_csv(KEY)
merged = key.merge(labels, on='item_no', how='inner')
print(f"Matched {len(merged)} rows")

# ── Save archival file ────────────────────────────────────────────────────────
merged.to_csv(OUT_CSV, index=False)
print(f"Saved to: {OUT_CSV}")

# ── Metrics ───────────────────────────────────────────────────────────────────
human = merged['human_score'].values
llm   = merged['llm_score'].values

# Overall accuracy
acc = (human == llm).mean()

# Per-class precision, recall, F1
classes = [-1, 0, 1]
class_names = {-1: 'Dovish (-1)', 0: 'Neutral (0)', 1: 'Hawkish (+1)'}
metrics = {}
for c in classes:
    tp = ((llm == c) & (human == c)).sum()
    fp = ((llm == c) & (human != c)).sum()
    fn = ((llm != c) & (human == c)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float('nan')
    metrics[c] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec, 'f1': f1}

# Macro F1
macro_f1 = np.nanmean([metrics[c]['f1'] for c in classes])

# Cohen's kappa
def cohen_kappa(a, b, labels):
    n = len(a)
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in labels)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

kappa = cohen_kappa(human, llm, classes)

# Confusion matrix
conf = pd.crosstab(
    pd.Series(human, name='Human'),
    pd.Series(llm,   name='LLM'),
    rownames=['Human label'],
    colnames=['LLM score']
)

# By-president breakdown
pres_metrics = {}
for pres in ['Macri', 'AF', 'Milei']:
    sub = merged[merged['president'] == pres]
    if len(sub) == 0:
        continue
    h = sub['human_score'].values
    l = sub['llm_score'].values
    pa = (h == l).mean()
    pf1s = []
    for c in classes:
        tp = ((l == c) & (h == c)).sum()
        fp = ((l == c) & (h != c)).sum()
        fn = ((l != c) & (h == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float('nan')
        pf1s.append(f1)
    pres_metrics[pres] = {
        'n': len(sub),
        'accuracy': pa,
        'macro_f1': np.nanmean(pf1s),
        'kappa': cohen_kappa(h, l, classes)
    }

# Disagreement case accuracy
dis  = merged[merged['disagrees'] == True]
agr  = merged[merged['disagrees'] == False]
dis_acc = (dis['human_score'] == dis['llm_score']).mean() if len(dis) > 0 else float('nan')
agr_acc = (agr['human_score'] == agr['llm_score']).mean() if len(agr) > 0 else float('nan')

# ── Report ────────────────────────────────────────────────────────────────────
lines = [
    "═" * 70,
    "  HUMAN VALIDATION REPORT — LLM FISCAL STANCE SCORING",
    "═" * 70,
    "",
    f"  Paragraphs labeled: {len(merged)}",
    f"  Presidents: Macri / AF / Milei  ({merged['president'].value_counts().to_dict()})",
    f"  LLM-dict disagreement cases: {merged['disagrees'].sum()} / {len(merged)}",
    "",
    "── OVERALL METRICS ─────────────────────────────────────────────────────",
    f"  Accuracy:   {acc:.3f}",
    f"  Macro F1:   {macro_f1:.3f}",
    f"  Cohen's κ:  {kappa:.3f}",
    "",
    "── PER-CLASS METRICS ───────────────────────────────────────────────────",
    f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N (human)':>10}",
    "  " + "-" * 50,
]
for c in classes:
    n_human = (human == c).sum()
    m = metrics[c]
    lines.append(
        f"  {class_names[c]:<14} {m['precision']:>10.3f} {m['recall']:>10.3f} "
        f"{m['f1']:>10.3f} {n_human:>10}"
    )

lines += [
    "",
    "── CONFUSION MATRIX (rows=human, cols=LLM) ─────────────────────────────",
]
lines.append("  " + conf.to_string().replace('\n', '\n  '))

lines += [
    "",
    "── BY-PRESIDENT BREAKDOWN ──────────────────────────────────────────────",
    f"  {'President':<10} {'N':>5} {'Accuracy':>10} {'Macro F1':>10} {'κ':>8}",
    "  " + "-" * 45,
]
for pres, m in pres_metrics.items():
    lines.append(
        f"  {pres:<10} {m['n']:>5} {m['accuracy']:>10.3f} "
        f"{m['macro_f1']:>10.3f} {m['kappa']:>8.3f}"
    )

lines += [
    "",
    "── DISAGREEMENT CASES ──────────────────────────────────────────────────",
    f"  LLM-dict disagreements (n={len(dis)}): accuracy = {dis_acc:.3f}",
    f"  LLM-dict agreements    (n={len(agr)}): accuracy = {agr_acc:.3f}",
    "",
    "── INTERPRETATION GUIDE ────────────────────────────────────────────────",
    "  κ > 0.80  → strong agreement (publication standard)",
    "  κ 0.60–0.80 → moderate-strong agreement (acceptable for ongoing work)",
    "  κ < 0.60  → weak agreement (needs investigation by class/president)",
    "  Macro F1 > 0.70 → defensible for a zero-shot model on political text",
    "═" * 70,
]

report = "\n".join(lines)
print("\n" + report)

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\nSaved: {OUT_TXT}")
