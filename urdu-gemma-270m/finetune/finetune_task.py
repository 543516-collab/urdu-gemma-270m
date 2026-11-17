"""
finetune_task.py
================
Generalized fine-tuner for all three downstream experiments.  It is
the same proven pipeline as finetune_lemma.py (16-bit LoRA, optional
CPT merge, prompt masking, 5 epochs, best-dev checkpoint), simply
parameterized by --task so one script covers every experiment.

    --task lemma  Experiment 3  (UDU, word/lemma)      -- already done
    --task pos    Experiment 2  (UMC, word/TAG)
    --task joint  Experiment 1  (UDU, word/TAG/lemma)

    --model base|cpt   base = fine-tune raw Gemma-3-270M
                       cpt  = fine-tune the CPT-adapted model

Per the thesis design, Experiment 2 (pos) and Experiment 1 (joint)
are run CPT-only, so you will normally use --model cpt for them.
The base option is kept available for completeness.

INPUT (JSONL under DRIVE_DIR), one object per line:
    {"prompt": "<instruction>: w1 w2 ...", "completion": "<labelled output>"}
The completion format is whatever the task requires:
    lemma : "w1/l1 w2/l2 ..."
    pos   : "w1/T1 w2/T2 ..."
    joint : "w1/T1/l1 w2/T2/l2 ..."
The script itself is format-agnostic -- it only trains the model to
produce the completion given the prompt -- so the labelled format is
entirely determined by the prepare_*.py scripts.

USAGE
-----
    python3 finetune_task.py --task pos   --model cpt
    python3 finetune_task.py --task joint --model cpt
    python3 finetune_task.py --task pos   --model cpt --micro-test

OUTPUT (on Drive)
-----------------
    {DRIVE_DIR}/{task}-{base|cpt}-lora-final/
"""

import argparse
import json
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from peft import PeftModel
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()

# ============================================================================
#  Google Drive persistence (robust: import-based Colab detection)
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
os.makedirs(DRIVE_DIR, exist_ok=True)


# ============================================================================
#  CONFIG
# ============================================================================
BASE_MODEL      = "unsloth/gemma-3-270m"
CPT_ADAPTER     = os.path.join(DRIVE_DIR, "urdu-gemma-270m-lora-final")

# Per-task data files (produced by the prepare_*.py scripts).
TASK_DATA = {
    "lemma": ("udu_lemma_train.jsonl", "udu_lemma_dev.jsonl"),
    "pos":   ("udu_pos_train.jsonl",   "udu_pos_dev.jsonl"),
    "joint": ("udu_joint_train.jsonl", "udu_joint_dev.jsonl"),
}

MAX_SEQ_LENGTH  = 512
SEPARATOR       = "\n"


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Gemma-3-270M for a downstream Urdu task"
    )
    p.add_argument("--task", choices=["lemma", "pos", "joint"], required=True,
                   help="lemma=Exp3, pos=Exp2, joint=Exp1")
    p.add_argument("--model", choices=["base", "cpt"], required=True,
                   help="'base'=raw Gemma-3-270M; 'cpt'=CPT-adapted model")
    p.add_argument("--micro-test", action="store_true", default=False,
                   help="Run only 50 steps as a sanity check")
    return p.parse_args()


# ============================================================================
#  Data
# ============================================================================
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_with_prompt_masking(examples, tokenizer, max_length):
    all_ids, all_mask, all_labels = [], [], []
    eos = tokenizer.eos_token or ""
    for ex in examples:
        p_enc = tokenizer(ex["prompt"] + SEPARATOR,
                          add_special_tokens=True, truncation=False)
        c_enc = tokenizer(ex["completion"] + eos,
                          add_special_tokens=False, truncation=False)
        input_ids = p_enc["input_ids"] + c_enc["input_ids"]
        attn      = p_enc["attention_mask"] + c_enc["attention_mask"]
        labels    = ([-100] * len(p_enc["input_ids"])) + c_enc["input_ids"]
        if len(input_ids) > max_length:
            input_ids, attn, labels = (input_ids[:max_length],
                                       attn[:max_length],
                                       labels[:max_length])
        all_ids.append(input_ids)
        all_mask.append(attn)
        all_labels.append(labels)
    return Dataset.from_dict({"input_ids": all_ids,
                              "attention_mask": all_mask,
                              "labels": all_labels})


class PromptMaskingCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = n - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


# ============================================================================
def main():
    args = parse_args()
    task, is_cpt, micro = args.task, args.model == "cpt", args.micro_test
    tag = "cpt" if is_cpt else "base"

    train_file, dev_file = TASK_DATA[task]
    train_path = os.path.join(DRIVE_DIR, train_file)
    dev_path   = os.path.join(DRIVE_DIR, dev_file)
    output_dir    = os.path.join(DRIVE_DIR, f"{task}-{tag}-checkpoints")
    final_adapter = os.path.join(DRIVE_DIR, f"{task}-{tag}-lora-final")

    for req in (train_path, dev_path):
        if not os.path.exists(req):
            raise FileNotFoundError(
                f"Expected input not found: {req}\n"
                f"Run the prepare script for the '{task}' task first, and "
                f"make sure its JSONL output is under DRIVE_DIR = {DRIVE_DIR}."
            )
    if is_cpt and not os.path.isdir(CPT_ADAPTER):
        raise FileNotFoundError(f"CPT adapter not found: {CPT_ADAPTER}")

    # ---- Load model (merge CPT for cpt mode, exactly as Exp 3) ----
    print("=" * 64)
    print(f"Task: {task}  |  Model: {tag}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=False,
    )
    if is_cpt:
        print(f"Attaching + merging CPT adapter from {CPT_ADAPTER} ...")
        model = PeftModel.from_pretrained(model, CPT_ADAPTER)
        model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Attaching FRESH task LoRA...")
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=42,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
    )

    # ---- Data ----
    print(f"Loading {train_path} ...")
    train_ex = load_jsonl(train_path)
    print(f"  {len(train_ex)} training examples")
    print(f"Loading {dev_path} ...")
    dev_ex = load_jsonl(dev_path)
    print(f"  {len(dev_ex)} dev examples")

    train_ds = tokenize_with_prompt_masking(train_ex, tokenizer, MAX_SEQ_LENGTH)
    dev_ds   = tokenize_with_prompt_masking(dev_ex, tokenizer, MAX_SEQ_LENGTH)

    ex0 = train_ds[0]
    print("\nSanity check (first example):")
    print(f"  Total tokens: {len(ex0['input_ids'])}")
    print(f"  Prompt tokens (masked): {sum(1 for l in ex0['labels'] if l == -100)}")
    comp_ids = [l for l in ex0["labels"] if l != -100]
    print(f"  Decoded completion: "
          f"{tokenizer.decode(comp_ids, skip_special_tokens=True)[:100]}...\n")

    # ---- Training args (identical to the Experiment 3 configuration) ----
    use_bf16 = is_bfloat16_supported()
    common = dict(
        output_dir=output_dir, per_device_train_batch_size=8,
        gradient_accumulation_steps=1, learning_rate=2e-4,
        max_grad_norm=1.0, bf16=use_bf16, fp16=False,
        optim="adamw_8bit", weight_decay=0.01, report_to="none",
    )
    if micro:
        print("*** MICRO-TEST MODE: 50 steps ***\n")
        targs = TrainingArguments(
            warmup_steps=10, max_steps=50, logging_steps=5,
            save_steps=50, save_total_limit=1,
            eval_strategy="steps", eval_steps=25, **common,
        )
    else:
        print("*** FULL FINE-TUNING MODE ***\n")
        targs = TrainingArguments(
            warmup_steps=50, num_train_epochs=5, logging_steps=10,
            save_steps=100, save_total_limit=2,
            eval_strategy="steps", eval_steps=100,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, **common,
        )

    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=PromptMaskingCollator(tokenizer))

    # ---- Resume (checkpoints persist on Drive) ----
    resume = os.path.isdir(output_dir) and any(
        "checkpoint" in d for d in os.listdir(output_dir))
    print("Found checkpoints on Drive, resuming..." if resume
          else "No checkpoints found. Starting fresh!")

    print(f"\nStarting {task} fine-tuning ({tag} model)...")
    trainer.train(resume_from_checkpoint=resume)

    print(f"\nSaving adapter to {final_adapter} ...")
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)

    print("\n" + "=" * 64)
    exp = {"lemma": "3", "pos": "2", "joint": "1"}[task]
    print(f"EXPERIMENT {exp} ({task}, {tag}) TRAINING COMPLETE")
    print(f"Adapter saved to: {final_adapter}")
    if micro:
        print("Micro-test done. If loss decreased, re-run without --micro-test.")
    else:
        print(f"Next: python3 evaluate_task.py --task {task} --model {tag}")
    print("=" * 64)


if __name__ == "__main__":
    main()
