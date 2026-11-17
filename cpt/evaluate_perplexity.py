"""
evaluate_perplexity.py
======================
Intrinsic evaluation (Section 4.6 / 6.1 of the thesis): perplexity of the base
Gemma-3-270M and of the CPT model (base + CPT LoRA adapter) on the UDU dev
split.  Each of the 552 sentences (`# text =` lines of the CoNLL-U file) is
scored on its own with the Gemma-3 tokenizer; perplexity = exp(mean token loss).

Reported: base 115.35, CPT 59.58 (-48.3%), over 21,818 scored tokens.

Usage (run once per model, each in a fresh process):
    python evaluate_perplexity.py base --dev ur_udtb-ud-dev.conllu
    python evaluate_perplexity.py cpt  --dev ur_udtb-ud-dev.conllu \
                                       --adapter urdu-gemma-270m-lora-final
"""
import argparse
import math

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


def load_udu_sentences(path):
    prefix = "# text = "
    with open(path, encoding="utf-8") as f:
        return [l[len(prefix):].strip() for l in (x.rstrip("\n") for x in f)
                if l.startswith(prefix)]


@torch.no_grad()
def compute_perplexity(model, tokenizer, sentences, max_length=2048):
    total_loss, total_tokens = 0.0, 0
    model.eval()
    for sent in sentences:
        if not sent:
            continue
        ids = tokenizer(sent, return_tensors="pt", truncation=True,
                        max_length=max_length)["input_ids"].to(model.device)
        if ids.shape[1] < 2:
            continue
        loss = model(input_ids=ids, labels=ids).loss
        n = ids.shape[1] - 1
        total_loss += loss.item() * n
        total_tokens += n
    avg = total_loss / total_tokens
    return avg, math.exp(avg), total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["base", "cpt"])
    ap.add_argument("--dev", default="ur_udtb-ud-dev.conllu")
    ap.add_argument("--adapter", default="urdu-gemma-270m-lora-final")
    ap.add_argument("--model", default="unsloth/gemma-3-270m")
    args = ap.parse_args()

    sentences = load_udu_sentences(args.dev)
    print(f"Loaded {len(sentences)} sentences")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=None, load_in_4bit=False)
    if args.which == "cpt":
        model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    avg, ppl, n = compute_perplexity(model, tokenizer, sentences)
    print(f"{args.which.upper()}: avg loss {avg:.4f} nats, "
          f"perplexity {ppl:.2f}, tokens scored {n:,}")


if __name__ == "__main__":
    main()
