"""
rescore_joint.py
================
Re-scores the SAVED joint predictions (eval_joint_cpt_predictions.jsonl)
using surface-word alignment instead of strict position alignment.

WHY
---
The joint task (word/TAG/lemma triples) is a harder generation target,
and the model sometimes emits a different NUMBER of triples than the
gold -- typically by splitting or merging a token (e.g. one gold word
"گمبھی" -> two predicted triples "گمبھی/.../گم" + "بھی/.../بھی").

With strict position-based scoring, a single such split shifts every
following token by one slot, so many individually-correct triples are
counted as wrong.  That deflates the joint accuracy and confounds
"genuine joint interference" with "token-count drift".

This script re-scores the same predictions by ALIGNING predicted
tokens to gold tokens on the surface word using a standard sequence
alignment (Needleman-Wunsch / edit distance over word sequences).
Only gold words that align 1:1 with a predicted word are scored for
tag/lemma; splits, merges, insertions, and deletions are reported
separately as alignment gaps.  This yields an alignment-robust
POS / lemma / joint accuracy to report ALONGSIDE the strict number.

No model, no GPU -- it only reads the JSONL you already have.

USAGE
-----
    python3 rescore_joint.py
    python3 rescore_joint.py --file /content/drive/MyDrive/urdu_lemma/eval_joint_cpt_predictions.jsonl

INPUT
-----
Each line of the predictions file has:
    {"idx": int, "pred_text": "<generated completion>", "score": {...}}
The gold completion is read from the joint test file so we can align
against the true tokens.
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

UD_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def parse_triples(text):
    """Parse 'w/TAG/l w/TAG/l ...' -> [(word, tag, lemma), ...], tag-anchored
    (peel word, then tag; remainder is lemma so a stray '/' is harmless)."""
    out = []
    for tok in text.split():
        if "/" in tok:
            w, rest = tok.split("/", 1)
            if "/" in rest:
                t, l = rest.split("/", 1)
            else:
                t, l = rest, ""
        else:
            w, t, l = tok, "", ""
        out.append((norm(w), norm(t), norm(l)))
    return out


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================================
#  Sequence alignment on the surface-word sequences (Needleman-Wunsch).
#  Returns a list of (gi, pj) aligned index pairs where both are present
#  (a 1:1 word match by position in the alignment); unmatched gold or
#  predicted tokens are gaps and are counted separately.
# ============================================================================
def align(gold_words, pred_words, match=2, mismatch=-1, gap=-1):
    n, m = len(gold_words), len(pred_words)
    # DP score + backpointer
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        bp[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        bp[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if gold_words[i - 1] == pred_words[j - 1] else mismatch
            diag = dp[i - 1][j - 1] + s
            up   = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = max(diag, up, left)
            dp[i][j] = best
            bp[i][j] = "diag" if best == diag else ("up" if best == up else "left")

    # traceback
    pairs = []            # (gi, pj) for aligned columns where both present
    gaps_gold = 0         # gold tokens with no predicted partner (deletions)
    gaps_pred = 0         # predicted tokens with no gold partner (insertions)
    i, j = n, m
    while i > 0 or j > 0:
        move = bp[i][j]
        if move == "diag":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "up":
            gaps_gold += 1
            i -= 1
        else:
            gaps_pred += 1
            j -= 1
    pairs.reverse()
    return pairs, gaps_gold, gaps_pred


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(
        DRIVE_DIR, "eval_joint_cpt_predictions.jsonl"))
    ap.add_argument("--test", default=os.path.join(
        DRIVE_DIR, "udu_joint_test.jsonl"))
    ap.add_argument("--require-word-match", action="store_true",
                    help="Only score aligned pairs whose surface words are "
                         "identical (strictest word-aligned variant).")
    args = ap.parse_args()

    preds = load_jsonl(args.file)
    gold_rows = load_jsonl(args.test)
    if len(preds) != len(gold_rows):
        print(f"NOTE: {len(preds)} predictions vs {len(gold_rows)} gold "
              f"sentences; scoring the first {min(len(preds), len(gold_rows))}.")

    # aggregate counters
    gold_tokens = 0
    aligned_pairs = 0
    pos_ok = lemma_ok = joint_ok = 0
    word_match_pairs = 0
    tot_gaps_gold = tot_gaps_pred = 0

    for pr in preds:
        gi_row = gold_rows[pr["idx"]] if pr["idx"] < len(gold_rows) else None
        if gi_row is None:
            continue
        gold = parse_triples(gi_row["completion"])
        pred = parse_triples(pr["pred_text"])
        gold_tokens += len(gold)

        gpairs, ggap, pgap = align([g[0] for g in gold],
                                   [p[0] for p in pred])
        tot_gaps_gold += ggap
        tot_gaps_pred += pgap

        for gi, pj in gpairs:
            gw, gt, gl = gold[gi]
            pw, pt, pl = pred[pj]
            same_word = (gw == pw)
            if same_word:
                word_match_pairs += 1
            if args.require_word_match and not same_word:
                continue
            aligned_pairs += 1
            tag_hit   = (pt == gt)
            lemma_hit = (pl == gl)
            pos_ok   += int(tag_hit)
            lemma_ok += int(lemma_hit)
            joint_ok += int(tag_hit and lemma_hit)

    denom = max(aligned_pairs, 1)
    print("=" * 64)
    print("ALIGNMENT-ROBUST RE-SCORING  (Experiment 1: joint, CPT)")
    print("=" * 64)
    print(f"  Gold tokens:                 {gold_tokens}")
    print(f"  Aligned & scored pairs:      {aligned_pairs}")
    print(f"  Surface-word match in align: "
          f"{100.0*word_match_pairs/max(aligned_pairs,1):.2f}%")
    print(f"  Unaligned gold (deletions):  {tot_gaps_gold}")
    print(f"  Unaligned pred (insertions): {tot_gaps_pred}  "
          f"(these are the token split/merge events)")
    print("  ---- accuracy over aligned tokens ----")
    print(f"  >>> POS accuracy:    {100.0*pos_ok/denom:.2f}%")
    print(f"  >>> Lemma accuracy:  {100.0*lemma_ok/denom:.2f}%")
    print(f"  >>> Joint accuracy:  {100.0*joint_ok/denom:.2f}%")
    print("=" * 64)
    print("  Interpretation:")
    print("   - Strict positional lemma acc was 89.77% (evaluate_task.py).")
    print("   - This word-aligned number removes the cascade penalty from")
    print("     token split/merge drift, so it reflects the model's actual")
    print("     tag/lemma quality on tokens it did emit in alignment.")
    print("   - The gap between the two = how much of the joint drop was")
    print("     alignment noise vs genuine interference.")
    print("   - Compare the aligned lemma acc against Experiment 3's 94.54%")
    print("     for the fairest joint-vs-separate read.")
    print("=" * 64)


if __name__ == "__main__":
    main()
