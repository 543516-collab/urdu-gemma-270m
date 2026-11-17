"""
rescore_pos_f1.py
=================
Computes per-tag precision / recall / F1, Macro-F1, and Weighted-F1
from the SAVED POS predictions (eval_pos_cpt_predictions.jsonl).

WHY
---
evaluate_task.py reports token ACCURACY (89.64%).  The closest
LLM-based Urdu POS benchmark, Arif et al. (2024) "Generalists vs.
Specialists" (EMNLP Findings), reports MACRO-F1 (their fine-tuned
Llama specialist: 67.55).  Accuracy and Macro-F1 are different
metrics, so the two numbers are not comparable.

Macro-F1 averages the per-tag F1 with EQUAL weight per tag, so rare
tags count as much as frequent ones -- a harder, more revealing
number.  Computing OUR Macro-F1 on the same UD benchmark makes the
comparison to Arif et al. apples-to-apples (same task, same data,
same metric).

No GPU, no model -- pure arithmetic over predictions you already have.

METHOD
------
Predictions are aligned to gold on the surface word (Needleman--Wunsch),
identical to rescore_joint.py, so token split/merge drift does not
distort the per-tag counts.  For each aligned token we record
(gold_tag, pred_tag).  From the resulting confusion counts:

    precision(t) = TP(t) / (TP(t) + FP(t))
    recall(t)    = TP(t) / (TP(t) + FN(t))
    F1(t)        = 2 * P * R / (P + R)
    Macro-F1     = mean over tags of F1(t)          (equal weight)
    Weighted-F1  = sum_t (support(t)/N) * F1(t)      (frequency weight)

USAGE
-----
    python3 rescore_pos_f1.py
    python3 rescore_pos_f1.py --file .../eval_pos_cpt_predictions.jsonl \
                              --test .../udu_pos_test.jsonl
"""

import argparse
import json
import os
import unicodedata


# ============================================================================
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB and not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")
DRIVE_DIR = os.environ.get("URDU_DIR", "/content/drive/MyDrive/urdu_lemma" if IN_COLAB else ".")


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def parse_pairs(text):
    """Parse 'w/TAG w/TAG ...' -> [(word, tag), ...] (word first, rest=tag)."""
    out = []
    for tok in text.split():
        if "/" in tok:
            w, t = tok.split("/", 1)
        else:
            w, t = tok, ""
        out.append((norm(w), norm(t)))
    return out


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def align(gw, pw, match=2, mismatch=-1, gap=-1):
    """Needleman-Wunsch over surface-word sequences; returns aligned
    (gi, pj) index pairs where both are present."""
    n, m = len(gw), len(pw)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bp = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap; bp[i][0] = "up"
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap; bp[0][j] = "left"
    for i in range(1, n+1):
        for j in range(1, m+1):
            s = match if gw[i-1] == pw[j-1] else mismatch
            diag, up, left = dp[i-1][j-1]+s, dp[i-1][j]+gap, dp[i][j-1]+gap
            best = max(diag, up, left)
            dp[i][j] = best
            bp[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        mv = bp[i][j]
        if mv == "diag":
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif mv == "up":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(
        DRIVE_DIR, "eval_pos_cpt_predictions.jsonl"))
    ap.add_argument("--test", default=os.path.join(
        DRIVE_DIR, "udu_pos_test.jsonl"))
    args = ap.parse_args()

    preds = load_jsonl(args.file)
    gold_rows = load_jsonl(args.test)

    # Confusion counts per tag.
    tp = {}   # correct predictions of tag t
    fp = {}   # predicted t but gold was something else
    fn = {}   # gold was t but predicted something else
    support = {}   # gold count of tag t
    tags = set()

    def bump(d, k):
        d[k] = d.get(k, 0) + 1

    aligned_pairs = 0
    for pr in preds:
        idx = pr["idx"]
        if idx >= len(gold_rows):
            continue
        gold = parse_pairs(gold_rows[idx]["completion"])
        pred = parse_pairs(pr["pred_text"])
        for gi, pj in align([g[0] for g in gold], [p[0] for p in pred]):
            gt = gold[gi][1]
            pt = pred[pj][1]
            tags.add(gt); tags.add(pt)
            bump(support, gt)
            aligned_pairs += 1
            if gt == pt:
                bump(tp, gt)
            else:
                bump(fp, pt)   # false positive for predicted tag
                bump(fn, gt)   # false negative for gold tag

    # Per-tag P/R/F1.
    def prf(t):
        TP, FP, FN = tp.get(t, 0), fp.get(t, 0), fn.get(t, 0)
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        F = 2*P*R/(P+R) if (P+R) else 0.0
        return P, R, F

    # Report per gold tag (tags that actually occur in the gold test set).
    gold_tags = sorted(support.keys())
    N = sum(support.values())

    print("=" * 68)
    print("PER-TAG POS METRICS  (Experiment 2, CPT model, UD test)")
    print("=" * 68)
    print(f"{'TAG':<8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Support':>9}")
    print("-" * 68)
    macro_sum = 0.0
    weighted_sum = 0.0
    for t in gold_tags:
        P, R, F = prf(t)
        macro_sum += F
        weighted_sum += (support[t] / N) * F
        print(f"{t:<8}{P*100:>8.2f}{R*100:>8.2f}{F*100:>8.2f}{support[t]:>9}")
    macro_f1 = macro_sum / len(gold_tags)
    weighted_f1 = weighted_sum
    micro_acc = sum(tp.get(t, 0) for t in gold_tags) / max(N, 1)

    print("-" * 68)
    print(f"{'Aligned tokens':<24}{aligned_pairs:>9}")
    print(f"{'Gold tags counted':<24}{len(gold_tags):>9}")
    print("=" * 68)
    print(f"  >>> Macro-F1:     {macro_f1*100:.2f}   "
          f"(equal weight per tag -- COMPARABLE to Arif et al. 67.55)")
    print(f"  >>> Weighted-F1:  {weighted_f1*100:.2f}   "
          f"(frequency-weighted)")
    print(f"      Token accuracy (aligned): {micro_acc*100:.2f}%   "
          f"(cf. evaluate_task.py's positional 89.64%)")
    print("=" * 68)
    print("  Interpretation:")
    print("   - Macro-F1 is the number to place beside Arif et al.'s 67.55")
    print("     (their fine-tuned Llama-3-8B specialist on UD POS).")
    print("   - It will typically sit BELOW token accuracy, because rare")
    print("     tags with few examples pull the unweighted average down.")
    print("   - The per-tag table above shows exactly which tags are strong")
    print("     or weak -- useful error-analysis material for the thesis.")
    print("=" * 68)


if __name__ == "__main__":
    main()
