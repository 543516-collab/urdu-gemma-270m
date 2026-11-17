"""
evaluate_lemma.py  (resumable edition)
======================================
Evaluates a fine-tuned Gemma-3-270M lemmatization model on the UDU
*test* split and reports word-level exact-match lemma accuracy.

This version is built to survive free-Colab timeouts:

  * RESUMABLE -- every sentence's result is appended to a JSONL file
    on Drive immediately (and flushed).  If the session dies, just
    re-run the exact same command: the script reads what's already
    on disk, skips those sentences, and continues from there.
  * CONSERVATIVE MEMORY -- generates one sentence at a time (batch
    size 1), so there is no risk of a batched out-of-memory crash on
    a small T4.
  * QUIET -- the repeated "max_new_tokens / max_length" warning is
    silenced so the log stays readable.

Two modes, matching finetune_lemma.py:
    --model base   Evaluate Experiment 3, base arm
    --model cpt    Evaluate Experiment 3, CPT arm

METRIC (unchanged)
------------------
Position-aligned, per-token exact-match on the LEMMA, NFC-normalized.
    accuracy = (# correct lemmas) / (# gold tokens)

USAGE
-----
    python3 evaluate_lemma.py --model base      # run; re-run if it times out
    python3 evaluate_lemma.py --model cpt

    # If you ever want to start over from scratch for a mode:
    python3 evaluate_lemma.py --model base --restart

OUTPUT
------
    Per-sentence predictions (also the resume file):
        {DRIVE_DIR}/eval_{base|cpt}_predictions.jsonl
    Final accuracy + diagnostics printed to the console once every
    test sentence has been processed.
"""

import argparse
import json
import os
import unicodedata
import torch

# Silence the per-call "max_new_tokens / max_length" warning so the log
# doesn't fill with it.  (Cosmetic only -- generation is unaffected.)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass


# ============================================================================
#  Google Drive persistence  (same robust detection as finetune_lemma.py)
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
BASE_MODEL      = "unsloth/gemma-3-270m"
CPT_ADAPTER     = os.path.join(DRIVE_DIR, "urdu-gemma-270m-lora-final")
TEST_DATA       = os.path.join(DRIVE_DIR, "udu_lemma_test.jsonl")

BASE_TASK_ADAPTER = os.path.join(DRIVE_DIR, "lemma-base-lora-final")
CPT_TASK_ADAPTER  = os.path.join(DRIVE_DIR, "lemma-cpt-lora-final")

MAX_SEQ_LENGTH  = 512     # must match training
SEPARATOR       = "\n"    # must match training


# ============================================================================
#  Argument parsing
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Gemma-3-270M Urdu lemmatization (resumable)"
    )
    p.add_argument("--model", choices=["base", "cpt"], required=True,
                   help="'base' = base arm; 'cpt' = CPT arm")
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Generation cap per sentence (default 384; "
                        "generation stops at EOS well before this for "
                        "normal sentences)")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Evaluate only the first N test sentences")
    p.add_argument("--restart", action="store_true",
                   help="Ignore any existing results file and start over")
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
    """Read an existing results file, tolerating a truncated final line
    (which can happen if the session was killed mid-write)."""
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
                    # truncated/corrupt final line -- ignore it
                    pass
    return rows


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def parse_pairs(text):
    pairs = []
    for tok in text.split():
        if "/" in tok:
            word, lemma = tok.rsplit("/", 1)
        else:
            word, lemma = tok, ""
        pairs.append((word, lemma))
    return pairs


def score_record(gold_pairs, pred_pairs):
    """Return (n_gold, n_correct, word_echo_ok, word_echo_tot)."""
    n_correct = echo_ok = echo_tot = 0
    for i, (gw, gl) in enumerate(gold_pairs):
        if i < len(pred_pairs):
            pw, pl = pred_pairs[i]
            echo_tot += 1
            if norm(pw) == norm(gw):
                echo_ok += 1
            if norm(pl) == norm(gl):
                n_correct += 1
    return len(gold_pairs), n_correct, echo_ok, echo_tot


# ============================================================================
#  Model loading -- rebuild exactly as finetune_lemma.py did
# ============================================================================
def load_model(mode):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    is_cpt = mode == "cpt"
    print("=" * 64)
    print(f"Loading model for Experiment 3{'b' if is_cpt else 'a'} ({mode})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    if is_cpt:
        if not os.path.isdir(CPT_ADAPTER):
            raise FileNotFoundError(
                f"CPT adapter not found: {CPT_ADAPTER}\n"
                f"Point CPT_ADAPTER at the folder containing "
                f"adapter_config.json + adapter_model.safetensors."
            )
        print(f"Attaching + merging CPT adapter from {CPT_ADAPTER} ...")
        model = PeftModel.from_pretrained(model, CPT_ADAPTER)
        model = model.merge_and_unload()

    task_adapter = CPT_TASK_ADAPTER if is_cpt else BASE_TASK_ADAPTER
    if not os.path.isdir(task_adapter):
        raise FileNotFoundError(
            f"Task adapter not found: {task_adapter}\n"
            f"Did finetune_lemma.py --model {mode} finish and save?"
        )
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
    enc = tokenizer(
        prompt + SEPARATOR,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ============================================================================
#  Main
# ============================================================================
def main():
    args = parse_args()
    mode = args.model

    if not os.path.exists(TEST_DATA):
        raise FileNotFoundError(
            f"Test set not found: {TEST_DATA}\n"
            f"Place udu_lemma_test.jsonl under DRIVE_DIR = {DRIVE_DIR}."
        )

    test = load_jsonl(TEST_DATA)
    if args.max_examples:
        test = test[: args.max_examples]
    total_test = len(test)
    print(f"Loaded {total_test} test sentences from {TEST_DATA}")

    out_path = os.path.join(DRIVE_DIR, f"eval_{mode}_predictions.jsonl")

    # ---- Resume: read what's already been done ----
    if args.restart and os.path.exists(out_path):
        os.remove(out_path)
        print("(--restart) Removed previous results; starting fresh.")

    done = load_existing(out_path)
    done_count = len(done)

    # Seed accumulators from already-completed sentences so final totals
    # are correct across sessions.
    total_tokens = correct_tokens = 0
    len_match = word_echo_ok = word_echo_tot = 0
    for r in done:
        total_tokens   += r.get("n_gold", 0)
        correct_tokens += r.get("n_correct", 0)
        if r.get("n_pred", -1) == r.get("n_gold", -2):
            len_match += 1
        word_echo_ok  += r.get("word_echo_ok", 0)
        word_echo_tot += r.get("word_echo_tot", 0)

    if done_count >= total_test:
        print(f"\nAll {total_test} sentences already evaluated. "
              f"Reporting from {out_path}.")
    else:
        if done_count > 0:
            print(f"Resuming: {done_count}/{total_test} already done, "
                  f"continuing with the rest.")
        model, tokenizer = load_model(mode)

        fout = open(out_path, "a", encoding="utf-8")
        print("\nEvaluating (greedy decoding, one sentence at a time)...")
        for idx in range(done_count, total_test):
            ex = test[idx]
            gold_pairs = parse_pairs(ex["completion"])
            pred_text  = generate_prediction(
                model, tokenizer, ex["prompt"], args.max_new_tokens
            )
            pred_pairs = parse_pairs(pred_text)

            n_gold, n_correct, echo_ok, echo_tot = score_record(
                gold_pairs, pred_pairs
            )
            total_tokens   += n_gold
            correct_tokens += n_correct
            if len(pred_pairs) == len(gold_pairs):
                len_match += 1
            word_echo_ok  += echo_ok
            word_echo_tot += echo_tot

            fout.write(json.dumps({
                "idx":           idx,
                "prompt":        ex["prompt"],
                "gold":          gold_pairs,
                "pred":          pred_pairs,
                "n_gold":        n_gold,
                "n_pred":        len(pred_pairs),
                "n_correct":     n_correct,
                "word_echo_ok":  echo_ok,
                "word_echo_tot": echo_tot,
            }, ensure_ascii=False) + "\n")
            fout.flush()               # persist immediately so a timeout
            os.fsync(fout.fileno())    # can't lose this sentence

            if (idx + 1) % 25 == 0:
                running = 100.0 * correct_tokens / max(total_tokens, 1)
                print(f"  {idx + 1}/{total_test} "
                      f"(running accuracy: {running:.2f}%)")

        fout.close()

    # ---- Final results (aggregated across all sessions) ----
    accuracy  = 100.0 * correct_tokens / max(total_tokens, 1)
    len_rate  = 100.0 * len_match / max(total_test, 1)
    echo_rate = 100.0 * word_echo_ok / max(word_echo_tot, 1)

    print("\n" + "=" * 64)
    print(f"RESULT  (Experiment 3{'b' if mode == 'cpt' else 'a'}: "
          f"{mode.upper()} model, UDU test split)")
    print("=" * 64)
    print(f"  Sentences evaluated:        {total_test}")
    print(f"  Gold tokens:                {total_tokens}")
    print(f"  Correct lemmas:             {correct_tokens}")
    print(f"  >>> Lemma accuracy:         {accuracy:.2f}%")
    print("  ---- diagnostics ----")
    print(f"  Length-match rate:          {len_rate:.2f}%")
    print(f"  Word-echo accuracy:         {echo_rate:.2f}%")
    print(f"  Per-sentence predictions:   {out_path}")
    print("=" * 64)
    print(f"  Baseline (Hafeez et al.):   96.30%")
    print("=" * 64)
    if len(load_existing(out_path)) < total_test:
        print("\nNot finished yet -- re-run the SAME command to continue "
              "from where it stopped.")
    else:
        print("\nDone. Record this accuracy for the results chapter.")


if __name__ == "__main__":
    main()
