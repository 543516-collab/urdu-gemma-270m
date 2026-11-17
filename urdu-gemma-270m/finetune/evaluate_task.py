"""
evaluate_task.py  (generalized, resumable)
==========================================
Evaluates a fine-tuned Gemma-3-270M model on the test split for any
of the three downstream experiments and reports exact-match accuracy.

    --task lemma  Experiment 3  (UDU, word/lemma)      -> lemma accuracy
    --task pos    Experiment 2  (UMC, word/TAG)        -> POS accuracy
    --task joint  Experiment 1  (UDU, word/TAG/lemma)  -> POS, lemma, joint

    --model base|cpt   rebuilds the model exactly as finetune_task.py did
                       (cpt merges the CPT adapter first, then attaches the
                        task adapter)

RESUMABLE + CONSERVATIVE MEMORY (identical machinery to the lemma
evaluator): one sentence at a time, each result flushed to a JSONL on
Drive, re-run the same command to continue after a timeout.

PARSING (robust)
----------------
Tokens are split with the POS tag as the anchor rather than by blindly
splitting on every slash:
    pos   : "word/TAG"        -> split once on "/"  (word, TAG)
    joint : "word/TAG/lemma"  -> peel word (split once), then peel TAG
                                 (split once); everything after is the
                                 lemma, so a stray "/" inside a lemma is
                                 harmless.
For pos/joint, the middle field is validated against the known UD tag
set (or the UMC/Sajjad tag set, configurable); a token whose tag field
is not a recognized tag is counted as a format error, not silently
mis-scored.

USAGE
-----
    python3 evaluate_task.py --task pos   --model cpt
    python3 evaluate_task.py --task joint --model cpt

OUTPUT
------
    {DRIVE_DIR}/eval_{task}_{base|cpt}_predictions.jsonl   (+ printed summary)
"""

import argparse
import json
import os
import unicodedata
import torch

try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

# Silence the live per-batch progress *bars* (the flickering "3% 2/69 ..."
# animation) that floods and freezes the Colab tab.  This does NOT remove
# the loss / eval_loss log lines -- those still print normally.
try:
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
except Exception:
    pass


# ============================================================================
#  Google Drive persistence
# ============================================================================
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB and not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

# >>> EDIT THIS ONE LINE to match your Google Drive folder <<<
DRIVE_DIR = os.environ.get("URDU_DIR", "/content/drive/MyDrive/urdu_lemma" if IN_COLAB else ".")


# ============================================================================
#  CONFIG
# ============================================================================
BASE_MODEL  = "unsloth/gemma-3-270m"
CPT_ADAPTER = os.path.join(DRIVE_DIR, "urdu-gemma-270m-lora-final")

# Per-task test file and how many "/"-separated fields each token has.
TASK_TEST = {
    "lemma": "udu_lemma_test.jsonl",
    "pos":   "udu_pos_test.jsonl",
    "joint": "udu_joint_test.jsonl",
}

# Known tag sets used to validate the tag field (and flag format errors).
# UD tags for the joint UDU task; extend UMC_TAGS with the exact Sajjad
# 42-tag inventory once the UMC prep is finalized.
UD_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}
# For the POS task the data is now the UDU (Universal Dependencies)
# treebank, so validate the tag field against the UD tag set (same as
# the joint task).  A predicted tag outside this set is flagged as a
# format error rather than silently mis-scored.
UMC_TAGS = UD_TAGS

MAX_SEQ_LENGTH = 512
SEPARATOR      = "\n"


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a downstream Urdu task")
    p.add_argument("--task", choices=["lemma", "pos", "joint"], required=True)
    p.add_argument("--model", choices=["base", "cpt"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--restart", action="store_true")
    return p.parse_args()


# ============================================================================
#  Helpers
# ============================================================================
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def parse_tokens(text, task, valid_tags):
    """
    Parse a completion string into per-word records.

    Returns a list of dicts, one per whitespace-separated token:
        lemma : {"word", "lemma"}
        pos   : {"word", "tag", "tag_ok"}
        joint : {"word", "tag", "lemma", "tag_ok"}
    'tag_ok' is False when the tag field is not in valid_tags (a format
    error), unless valid_tags is None (validation disabled).
    """
    out = []
    for tok in text.split():
        if task == "lemma":
            if "/" in tok:
                w, l = tok.rsplit("/", 1)
            else:
                w, l = tok, ""
            out.append({"word": w, "lemma": l})

        elif task == "pos":
            if "/" in tok:
                w, t = tok.split("/", 1)   # word first, rest is tag
            else:
                w, t = tok, ""
            tag_ok = (valid_tags is None) or (t in valid_tags)
            out.append({"word": w, "tag": t, "tag_ok": tag_ok})

        else:  # joint: word/TAG/lemma, tag-anchored
            # peel the word off first, then the tag; remainder is the lemma
            if "/" in tok:
                w, rest = tok.split("/", 1)
                if "/" in rest:
                    t, l = rest.split("/", 1)
                else:
                    t, l = rest, ""
            else:
                w, t, l = tok, "", ""
            tag_ok = (valid_tags is None) or (t in valid_tags)
            out.append({"word": w, "tag": t, "lemma": l, "tag_ok": tag_ok})
    return out


# ============================================================================
#  Model loading (rebuild exactly as finetune_task.py did)
# ============================================================================
def load_model(task, mode):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    is_cpt = mode == "cpt"
    tag = "cpt" if is_cpt else "base"
    task_adapter = os.path.join(DRIVE_DIR, f"{task}-{tag}-lora-final")

    print("=" * 64)
    print(f"Loading model  |  task={task}  model={tag}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=False,
    )
    if is_cpt:
        if not os.path.isdir(CPT_ADAPTER):
            raise FileNotFoundError(f"CPT adapter not found: {CPT_ADAPTER}")
        print(f"Attaching + merging CPT adapter from {CPT_ADAPTER} ...")
        model = PeftModel.from_pretrained(model, CPT_ADAPTER)
        model = model.merge_and_unload()

    if not os.path.isdir(task_adapter):
        raise FileNotFoundError(
            f"Task adapter not found: {task_adapter}\n"
            f"Did finetune_task.py --task {task} --model {tag} finish?")
    print(f"Attaching task adapter from {task_adapter} ...")
    model = PeftModel.from_pretrained(model, task_adapter)

    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_prediction(model, tokenizer, prompt, max_new_tokens):
    enc = tokenizer(prompt + SEPARATOR, return_tensors="pt",
                    add_special_tokens=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)


# ============================================================================
#  Per-sentence scoring
# ============================================================================
def score_sentence(task, gold, pred, train_vocab=None):
    """
    Compare gold and predicted token lists (from parse_tokens).
    Returns a dict of per-sentence counts, aggregated by the caller.
    'train_vocab' (a set of surface words seen in training) enables the
    known/unknown-word breakdown for the pos task.
    """
    n = len(gold)
    r = {"n_gold": n, "pos_ok": 0, "lemma_ok": 0, "joint_ok": 0,
         "word_echo_ok": 0, "fmt_err": 0,
         "known_tot": 0, "known_ok": 0, "unknown_tot": 0, "unknown_ok": 0}
    for i, g in enumerate(gold):
        p = pred[i] if i < len(pred) else None
        if p is None:
            continue
        if norm(p["word"]) == norm(g["word"]):
            r["word_echo_ok"] += 1

        if task == "lemma":
            if norm(p["lemma"]) == norm(g["lemma"]):
                r["lemma_ok"] += 1

        elif task == "pos":
            if not p.get("tag_ok", True):
                r["fmt_err"] += 1
            tag_hit = norm(p["tag"]) == norm(g["tag"])
            if tag_hit:
                r["pos_ok"] += 1
            if train_vocab is not None:
                known = norm(g["word"]) in train_vocab
                if known:
                    r["known_tot"] += 1
                    r["known_ok"] += int(tag_hit)
                else:
                    r["unknown_tot"] += 1
                    r["unknown_ok"] += int(tag_hit)

        else:  # joint
            if not p.get("tag_ok", True):
                r["fmt_err"] += 1
            tag_hit   = norm(p["tag"])   == norm(g["tag"])
            lemma_hit = norm(p["lemma"]) == norm(g["lemma"])
            r["pos_ok"]   += int(tag_hit)
            r["lemma_ok"] += int(lemma_hit)
            r["joint_ok"] += int(tag_hit and lemma_hit)
    r["len_match"] = int(len(pred) == n)
    return r


def build_train_vocab(task):
    """For the pos task, collect surface words seen in the UMC training
    split to drive the known/unknown breakdown.  Returns None if the
    training file is unavailable."""
    if task != "pos":
        return None
    train_path = os.path.join(DRIVE_DIR, "udu_pos_train.jsonl")
    if not os.path.exists(train_path):
        return None
    vocab = set()
    for ex in load_jsonl(train_path):
        for tok in parse_tokens(ex["completion"], "pos", None):
            vocab.add(norm(tok["word"]))
    return vocab


# ============================================================================
def main():
    args = parse_args()
    task, mode = args.task, args.model
    valid_tags = UD_TAGS if task == "joint" else (UMC_TAGS if task == "pos"
                                                  else None)

    test_path = os.path.join(DRIVE_DIR, TASK_TEST[task])
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found: {test_path}")
    test = load_jsonl(test_path)
    if args.max_examples:
        test = test[: args.max_examples]
    total = len(test)
    print(f"Loaded {total} test sentences from {test_path}")

    train_vocab = build_train_vocab(task)
    if task == "pos":
        print("Known/unknown breakdown: "
              + ("enabled" if train_vocab is not None
                 else "disabled (udu_pos_train.jsonl not found)"))

    out_path = os.path.join(DRIVE_DIR, f"eval_{task}_{mode}_predictions.jsonl")
    if args.restart and os.path.exists(out_path):
        os.remove(out_path)
        print("(--restart) previous results removed.")

    done = load_existing(out_path)
    agg = {k: 0 for k in ["n_gold", "pos_ok", "lemma_ok", "joint_ok",
                          "word_echo_ok", "fmt_err", "len_match",
                          "known_tot", "known_ok", "unknown_tot",
                          "unknown_ok"]}
    for rec in done:
        s = rec["score"]
        for k in agg:
            agg[k] += s.get(k, 0)
    done_count = len(done)

    if done_count >= total:
        print(f"\nAll {total} sentences already evaluated.")
    else:
        if done_count:
            print(f"Resuming: {done_count}/{total} already done.")
        model, tokenizer = load_model(task, mode)
        fout = open(out_path, "a", encoding="utf-8")
        print("\nEvaluating (greedy, one sentence at a time)...")
        for idx in range(done_count, total):
            ex = test[idx]
            gold = parse_tokens(ex["completion"], task, valid_tags)
            pred_text = generate_prediction(model, tokenizer,
                                            ex["prompt"], args.max_new_tokens)
            pred = parse_tokens(pred_text, task, valid_tags)
            s = score_sentence(task, gold, pred, train_vocab)
            for k in agg:
                agg[k] += s.get(k, 0)
            fout.write(json.dumps({"idx": idx, "pred_text": pred_text,
                                   "score": s}, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            if (idx + 1) % 25 == 0:
                key = "lemma_ok" if task == "lemma" else "pos_ok"
                run = 100.0 * agg[key] / max(agg["n_gold"], 1)
                print(f"  {idx + 1}/{total} (running "
                      f"{'lemma' if task == 'lemma' else 'POS'} acc: "
                      f"{run:.2f}%)")
        fout.close()

    # ---- Final report ----
    tot = max(agg["n_gold"], 1)
    print("\n" + "=" * 64)
    exp = {"lemma": "3", "pos": "2", "joint": "1"}[task]
    print(f"RESULT  (Experiment {exp}: {task}, {mode.upper()} model)")
    print("=" * 64)
    print(f"  Sentences: {total}    Gold tokens: {agg['n_gold']}")
    if task == "lemma":
        print(f"  >>> Lemma accuracy:  {100.0*agg['lemma_ok']/tot:.2f}%")
    elif task == "pos":
        print(f"  >>> POS accuracy:    {100.0*agg['pos_ok']/tot:.2f}%")
        if agg["known_tot"] or agg["unknown_tot"]:
            kt, ut = max(agg["known_tot"], 1), max(agg["unknown_tot"], 1)
            print(f"      known-word POS:   {100.0*agg['known_ok']/kt:.2f}% "
                  f"({agg['known_tot']} tokens)")
            print(f"      unknown-word POS: {100.0*agg['unknown_ok']/ut:.2f}% "
                  f"({agg['unknown_tot']} tokens)")
        print(f"      (Jawaid et al. baseline: 88.74% overall, "
              f"54.76% unknown)")
    else:  # joint
        print(f"  >>> POS accuracy:    {100.0*agg['pos_ok']/tot:.2f}%")
        print(f"  >>> Lemma accuracy:  {100.0*agg['lemma_ok']/tot:.2f}%")
        print(f"  >>> Joint accuracy:  {100.0*agg['joint_ok']/tot:.2f}%  "
              f"(both correct on the same token)")
        print(f"      (compare lemma vs Experiment 3's 94.54%)")
    print("  ---- diagnostics ----")
    print(f"  Length-match rate:  {100.0*agg['len_match']/max(total,1):.2f}%")
    print(f"  Word-echo accuracy: {100.0*agg['word_echo_ok']/tot:.2f}%")
    if task in ("pos", "joint"):
        print(f"  Tag format errors:  {agg['fmt_err']} tokens")
    print(f"  Per-sentence file:  {out_path}")
    print("=" * 64)
    if len(load_existing(out_path)) < total:
        print("\nNot finished -- re-run the SAME command to continue.")
    else:
        print("\nDone. Record these numbers for the results chapter.")


if __name__ == "__main__":
    main()
