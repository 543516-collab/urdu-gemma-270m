"""
tokenize_and_pack.py
====================
Tokenize the cleaned Urdu corpus with the Gemma-3-270M tokenizer and pack it
into fixed-length 2048-token blocks for continued pre-training (Section 3.4 /
4.3 of the thesis).

Input : JSONL files with one document per line, {"text": "..."}.
Output: a Hugging Face `datasets` folder (Arrow shards) with input_ids,
        attention_mask and labels.

Packing: documents are concatenated with an EOS token after each one and cut
into MAX_LEN blocks; the remainder of each map batch that does not fill a
whole block is dropped.  This is why the packed token count (748,086 x 2048 =
1,532,080,128) differs from the Unitok count reported for the corpus.

Usage:
    export HF_TOKEN=...            # if the model requires accepting a licence
    python tokenize_and_pack.py --input "ready_for_training/*.jsonl" \
                                --output tokenized_packed_2048_270m
"""
import argparse
import os

from datasets import Features, Value, load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of raw JSONL files")
    ap.add_argument("--output", default="tokenized_packed_2048_270m")
    ap.add_argument("--model-id", default="google/gemma-3-270m")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--num-proc", type=int, default=2)
    args = ap.parse_args()
    MAX_LEN = args.max_len

    raw = load_dataset("json", data_files=args.input, split="train",
                       features=Features({"text": Value("string")}))
    print(raw)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=os.environ.get("HF_TOKEN"), use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id.")

    # Round-trip sanity check on a known Urdu string before tokenizing ~1.4B tokens.
    test = "یہ ایک اردو جملہ ہے"
    if test not in tokenizer.decode(tokenizer.encode(test)):
        raise RuntimeError("Tokenizer round-trip check failed; check --model-id.")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=False)

    tokenized = raw.map(tokenize_fn, batched=True, num_proc=args.num_proc,
                        remove_columns=["text"], desc="Tokenizing",
                        writer_batch_size=2000)

    def group_texts(examples):
        ids, attn = [], []
        for i, a in zip(examples["input_ids"], examples["attention_mask"]):
            ids.extend(i); attn.extend(a)
            ids.append(eos_id); attn.append(1)
        total = (len(ids) // MAX_LEN) * MAX_LEN
        input_ids = [ids[i:i + MAX_LEN] for i in range(0, total, MAX_LEN)]
        attention = [attn[i:i + MAX_LEN] for i in range(0, total, MAX_LEN)]
        return {"input_ids": input_ids, "attention_mask": attention,
                "labels": [x.copy() for x in input_ids]}

    packed = tokenized.map(group_texts, batched=True, batch_size=1000,
                           num_proc=args.num_proc, desc=f"Packing to {MAX_LEN}",
                           writer_batch_size=2000)
    print(packed)
    os.makedirs(args.output, exist_ok=True)
    packed.save_to_disk(args.output)
    print("Saved to", args.output)

    # Decode a few packed blocks: they must read as Urdu.
    for i in range(min(3, len(packed))):
        print(f"\nSequence {i}:",
              tokenizer.decode(packed[i]["input_ids"][:60], skip_special_tokens=True))


if __name__ == "__main__":
    main()
