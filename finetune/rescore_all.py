"""
rescore_all.py
==============
Re-scores the released predictions of every experiment arm and prints the
word-aligned and full-denominator figures reported in the thesis.

Alignment: Needleman-Wunsch over surface-word sequences (match 2, mismatch -1,
gap -1), as in rescore_joint.py / rescore_pos_f1.py.  Two rules are reported:

  all-pairs  : every aligned (gold, predicted) pair is scored; only gold words
               without a partner are unaligned.  Used for Experiments 1 and 2.
  same-word  : only aligned pairs whose surface words are identical are scored.
               Used for the Experiment 3 word-aligned column (96.25 / 96.34).

"full" divides the number of correct aligned predictions by all 14,806 gold
tokens, i.e. counts unaligned gold tokens as errors (all-pairs rule).

Usage:
    python rescore_all.py --data ../data --pred ../results/predictions
"""
import argparse
import json
import os
import unicodedata


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def align(gw, pw, match=2, mismatch=-1, gap=-1):
    n, m = len(gw), len(pw)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], bp[i][0] = dp[i - 1][0] + gap, "up"
    for j in range(1, m + 1):
        dp[0][j], bp[0][j] = dp[0][j - 1] + gap, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if gw[i - 1] == pw[j - 1] else mismatch
            d, u, l = dp[i - 1][j - 1] + s, dp[i - 1][j] + gap, dp[i][j - 1] + gap
            best = max(d, u, l)
            dp[i][j] = best
            bp[i][j] = "diag" if best == d else ("up" if best == u else "left")
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        mv = bp[i][j]
        if mv == "diag":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif mv == "up":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def parse(text, task):
    out = []
    for tok in text.split():
        if task == "joint":
            if "/" in tok:
                w, rest = tok.split("/", 1)
                t, l = rest.split("/", 1) if "/" in rest else (rest, "")
            else:
                w, t, l = tok, "", ""
            out.append((norm(w), norm(t), norm(l)))
        else:
            if "/" in tok:
                w, v = tok.rsplit("/", 1) if task == "lemma" else tok.split("/", 1)
            else:
                w, v = tok, ""
            out.append((norm(w), norm(v)))
    return out


def pred_text(rec):
    if "pred_text" in rec:
        return rec["pred_text"]
    p = rec["pred"]  # evaluate_lemma.py stores a list of [word, lemma] pairs
    return " ".join(x if isinstance(x, str) else "/".join(x) for x in p)


ARMS = [
    ("Exp. 3 lemma, base arm", "eval_base_predictions.jsonl", "udu_lemma_test.jsonl", "lemma"),
    ("Exp. 3 lemma, CPT arm", "eval_cpt_predictions.jsonl", "udu_lemma_test.jsonl", "lemma"),
    ("Exp. 2 POS, base arm", "eval_pos_base_predictions.jsonl", "udu_pos_test.jsonl", "pos"),
    ("Exp. 2 POS, CPT arm", "eval_pos_cpt_predictions.jsonl", "udu_pos_test.jsonl", "pos"),
    ("Exp. 1 joint, CPT", "eval_joint_cpt_predictions.jsonl", "udu_joint_test.jsonl", "joint"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--pred", default="../results/predictions")
    args = ap.parse_args()

    for name, pf, gf, task in ARMS:
        gold_rows = load(os.path.join(args.data, gf))
        preds = load(os.path.join(args.pred, pf))
        cols = 2 if task == "joint" else 1  # joint: POS and lemma
        tot = mism = all_pairs = same_pairs = 0
        ok_all = [0] * cols
        ok_same = [0] * cols
        for rec in preds:
            g = parse(gold_rows[rec["idx"]]["completion"], task)
            p = parse(pred_text(rec), task)
            tot += len(g)
            mism += len(g) != len(p)
            for gi, pj in align([x[0] for x in g], [x[0] for x in p]):
                same = g[gi][0] == p[pj][0]
                all_pairs += 1
                same_pairs += same
                for c in range(cols):
                    hit = g[gi][1 + c] == p[pj][1 + c]
                    ok_all[c] += hit
                    ok_same[c] += hit and same
        labels = ["POS", "lemma"] if task == "joint" else [task]
        print(f"\n{name}: {tot} gold tokens, {mism} length-mismatched sentences")
        print(f"  all-pairs: {all_pairs} aligned ({tot - all_pairs} unaligned gold)")
        print(f"  same-word: {same_pairs} aligned ({tot - same_pairs} unscored gold)")
        for c, lab in enumerate(labels):
            print(f"  {lab:5s} all-pairs {100 * ok_all[c] / all_pairs:6.2f}   "
                  f"same-word {100 * ok_same[c] / same_pairs:6.2f}   "
                  f"full {100 * ok_all[c] / tot:6.2f}")


if __name__ == "__main__":
    main()
