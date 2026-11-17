"""
finetune_lemma.py
==================
Fine-tunes Gemma-3-270M for Urdu lemmatization (Experiment 3).

Supports two modes:
    --model base   Fine-tune the raw base model (Experiment 3, base arm)
    --model cpt    Fine-tune the CPT-adapted model (Experiment 3, CPT arm)

The comparison between the base and CPT arms directly answers:
"Did continued pre-training on 1.45B Urdu tokens improve
downstream lemmatization accuracy?"

DATA
----
Reads JSONL files produced by prepare_udu_lemma.py:
    udu_lemma_train.jsonl   (training set, ~4000 sentences)
    udu_lemma_dev.jsonl     (dev set, ~550 sentences)

Each line:  {"prompt": "Lemmatize: w1 w2...", "completion": "w1/l1 w2/l2..."}

The model learns to produce the completion given the prompt.  Loss is
computed ONLY on the completion tokens (the prompt tokens have their
labels set to -100 so they don't contribute to the loss).

PERSISTENCE (free Colab)
------------------------
On free Colab, /content is wiped when the session disconnects.  All
inputs and outputs therefore live on Google Drive under DRIVE_DIR
(set below).  Because the CHECKPOINTS are on Drive, a mid-run
disconnect is recoverable: just re-run the same command and training
resumes from the latest checkpoint automatically.

USAGE
-----
    # Experiment 3, base arm:
    python3 finetune_lemma.py --model base

    # Experiment 3, CPT arm:
    python3 finetune_lemma.py --model cpt

OUTPUT (on Drive)
-----------------
    {DRIVE_DIR}/lemma-base-lora-final/   (Experiment 3 base-arm adapter)
    {DRIVE_DIR}/lemma-cpt-lora-final/    (Experiment 3 CPT-arm adapter)
"""

import argparse
import json
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from peft import PeftModel


# ============================================================================
#  Google Drive persistence
#  (free Colab wipes /content on disconnect, so everything that must
#   survive -- inputs, checkpoints, final adapters -- lives on Drive)
# ============================================================================
# Detect Colab by whether the module imports -- NOT by whether mount()
# succeeds.  (If Drive is already mounted, a second mount() call can raise,
# and we must not let that flip us into "not in Colab" mode, which would
# send every path to the local, ephemeral /content directory.)
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    # Not running in Colab (e.g. a local machine): use a local directory.
    IN_COLAB = False

# Mount Drive only if it isn't already mounted.  Safe to run either way.
if IN_COLAB and not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

# >>> EDIT THIS ONE LINE to match the folder in your Google Drive <<<
# Everything the run reads and writes is resolved relative to here.
DRIVE_DIR = os.environ.get("URDU_DIR", "/content/drive/MyDrive/urdu_lemma" if IN_COLAB else ".")

# Make sure the base directory exists (harmless if it already does).
os.makedirs(DRIVE_DIR, exist_ok=True)


# ============================================================================
#  CONFIG
# ============================================================================
BASE_MODEL      = "unsloth/gemma-3-270m"

# Inputs -- these must already be present under DRIVE_DIR before you run.
CPT_ADAPTER     = os.path.join(DRIVE_DIR, "urdu-gemma-270m-lora-final")
TRAIN_DATA      = os.path.join(DRIVE_DIR, "udu_lemma_train.jsonl")
DEV_DATA        = os.path.join(DRIVE_DIR, "udu_lemma_dev.jsonl")

MAX_SEQ_LENGTH  = 512    # UDU sentences are short; 512 is more than enough
                          # and avoids the huge-vocabulary lm_head memory
                          # cost at 2048 that we profiled earlier

SEPARATOR = "\n"          # separates prompt from completion in the
                          # concatenated training string


# ============================================================================
#  Argument parsing
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma-3-270M for Urdu lemmatization"
    )
    parser.add_argument(
        "--model",
        choices=["base", "cpt"],
        required=True,
        help="'base' = fine-tune raw Gemma-3-270M; "
             "'cpt' = fine-tune CPT-adapted Gemma-3-270M"
    )
    parser.add_argument(
        "--micro-test",
        action="store_true",
        default=False,
        help="Run only 50 steps as a sanity check before committing"
    )
    return parser.parse_args()


# ============================================================================
#  Data loading and tokenization with prompt masking
# ============================================================================
def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def tokenize_with_prompt_masking(examples, tokenizer, max_length):
    """
    Tokenize prompt/completion pairs with prompt masking.

    For each example:
        full_text = prompt + SEPARATOR + completion + eos_token

    The labels are set to -100 for all prompt+separator tokens, so the
    model is only trained to predict the completion and the EOS.

    Returns a HuggingFace Dataset ready for the Trainer.
    """
    all_input_ids      = []
    all_attention_masks = []
    all_labels          = []

    eos_token = tokenizer.eos_token or ""

    for ex in examples:
        prompt     = ex["prompt"]
        completion = ex["completion"]

        # Tokenize prompt and completion separately to find the boundary
        prompt_text     = prompt + SEPARATOR
        completion_text = completion + eos_token

        prompt_enc = tokenizer(
            prompt_text,
            add_special_tokens=True,   # adds BOS at the start
            truncation=False,
        )
        completion_enc = tokenizer(
            completion_text,
            add_special_tokens=False,  # no extra BOS in the middle
            truncation=False,
        )

        # Concatenate
        input_ids = prompt_enc["input_ids"] + completion_enc["input_ids"]
        attn_mask = prompt_enc["attention_mask"] + completion_enc["attention_mask"]

        # Build labels: -100 for prompt tokens, actual IDs for completion
        prompt_len = len(prompt_enc["input_ids"])
        labels = ([-100] * prompt_len) + completion_enc["input_ids"]

        # Truncate if too long (shouldn't happen for UDU, but safety)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attn_mask = attn_mask[:max_length]
            labels    = labels[:max_length]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attn_mask)
        all_labels.append(labels)

    return Dataset.from_dict({
        "input_ids":      all_input_ids,
        "attention_mask":  all_attention_masks,
        "labels":          all_labels,
    })


# ============================================================================
#  Custom data collator (pads each batch to the same length)
# ============================================================================
class PromptMaskingCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence
    in the batch.  Padding tokens use:
        input_ids:      pad_token_id
        attention_mask: 0
        labels:         -100  (so padding doesn't contribute to loss)
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id  # fallback

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        batch = {
            "input_ids":      [],
            "attention_mask":  [],
            "labels":          [],
        }
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(
                f["input_ids"] + [self.pad_id] * pad_len
            )
            batch["attention_mask"].append(
                f["attention_mask"] + [0] * pad_len
            )
            batch["labels"].append(
                f["labels"] + [-100] * pad_len
            )

        return {k: torch.tensor(v) for k, v in batch.items()}


# ============================================================================
#  Main
# ============================================================================
def main():
    args = parse_args()
    is_cpt = args.model == "cpt"
    micro_test = args.micro_test

    # Output directory and adapter name depend on which model variant.
    # Both live on Drive so they survive a Colab disconnect (and so the
    # resume logic below can find the checkpoints again).
    if is_cpt:
        output_dir     = os.path.join(DRIVE_DIR, "lemma-cpt-checkpoints")
        final_adapter  = os.path.join(DRIVE_DIR, "lemma-cpt-lora-final")
    else:
        output_dir     = os.path.join(DRIVE_DIR, "lemma-base-checkpoints")
        final_adapter  = os.path.join(DRIVE_DIR, "lemma-base-lora-final")

    # Fail early with a clear message if the inputs aren't on Drive yet.
    for required in (TRAIN_DATA, DEV_DATA):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"Expected input not found: {required}\n"
                f"Make sure your data and CPT adapter live under "
                f"DRIVE_DIR = {DRIVE_DIR} (edit that line if your Drive "
                f"folder is named differently)."
            )
    if is_cpt and not os.path.exists(CPT_ADAPTER):
        raise FileNotFoundError(
            f"CPT adapter not found: {CPT_ADAPTER}\n"
            f"Copy your CPT adapter folder into DRIVE_DIR, or adjust "
            f"CPT_ADAPTER to point at its location on Drive."
        )

    # ---- 1. Load model ----
    print("=" * 64)
    if is_cpt:
        print("Loading BASE model + merging CPT adapter...")
    else:
        print("Loading BASE Gemma-3-270M (no CPT)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    if is_cpt:
        print(f"Attaching CPT adapter from {CPT_ADAPTER} ...")
        model = PeftModel.from_pretrained(model, CPT_ADAPTER)
        print("Merging CPT adapter into base weights...")
        model = model.merge_and_unload()
        print("CPT adapter merged. Model is now a plain (non-LoRA) model")
        print("with Urdu-adapted weights baked in.\n")

    # Ensure pad token is set (Gemma uses eos as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Attach fresh task-specific LoRA ----
    print("Attaching FRESH task LoRA for lemmatization...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,  # small dataset, no need
        random_state=42,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # Print trainable params for verification
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)} tensors")

    # ---- 3. Load and tokenize data ----
    print(f"\nLoading training data from {TRAIN_DATA} ...")
    train_examples = load_jsonl(TRAIN_DATA)
    print(f"Loaded {len(train_examples)} training examples")

    print(f"Loading dev data from {DEV_DATA} ...")
    dev_examples = load_jsonl(DEV_DATA)
    print(f"Loaded {len(dev_examples)} dev examples")

    print("\nTokenizing with prompt masking...")
    train_dataset = tokenize_with_prompt_masking(
        train_examples, tokenizer, MAX_SEQ_LENGTH
    )
    dev_dataset = tokenize_with_prompt_masking(
        dev_examples, tokenizer, MAX_SEQ_LENGTH
    )

    # Sanity check: decode one example to verify prompt masking
    ex = train_dataset[0]
    prompt_tokens = sum(1 for l in ex["labels"] if l == -100)
    completion_tokens = sum(1 for l in ex["labels"] if l != -100)
    print(f"\nSanity check (first training example):")
    print(f"  Total tokens: {len(ex['input_ids'])}")
    print(f"  Prompt tokens (masked, labels=-100): {prompt_tokens}")
    print(f"  Completion tokens (trained on): {completion_tokens}")

    # Decode just the completion part to verify it's correct
    completion_ids = [l for l in ex["labels"] if l != -100]
    decoded_completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    print(f"  Decoded completion: {decoded_completion[:100]}...")
    print()

    # ---- 4. Training setup ----
    use_bf16 = is_bfloat16_supported()

    if micro_test:
        print("*** MICRO-TEST MODE: 50 steps ***\n")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            max_steps=50,
            learning_rate=2e-4,           # standard LoRA task fine-tuning LR
            max_grad_norm=1.0,
            bf16=use_bf16,
            fp16=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=25,
            report_to="none",
        )
    else:
        print("*** FULL FINE-TUNING MODE ***\n")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,  # small dataset, no need to
                                             # accumulate
            warmup_steps=50,
            num_train_epochs=5,             # multiple epochs: UDU train is
                                             # only ~4000 sentences, so one
                                             # epoch barely touches the LoRA
                                             # parameters; 5 epochs gives
                                             # the adapter enough passes to
                                             # learn the lemmatization
                                             # patterns without overfitting
                                             # (monitored via dev eval)
            learning_rate=2e-4,             # standard LoRA task FT rate;
                                             # higher than CPT rate because
                                             # task FT operates on a small,
                                             # clean, curated dataset where
                                             # the signal-to-noise ratio is
                                             # much higher than web crawl
            max_grad_norm=1.0,
            bf16=use_bf16,
            fp16=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=100,                 # evaluate on dev set every 100
                                             # steps -- watch for dev loss
                                             # rising while train loss falls
                                             # (sign of overfitting)
            load_best_model_at_end=True,    # at the end of training, revert
                                             # to the checkpoint with the
                                             # lowest dev loss rather than
                                             # the final step -- protects
                                             # against overfitting on this
                                             # tiny dataset
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=PromptMaskingCollator(tokenizer),
    )

    # ---- 5. Resume logic ----
    # Because output_dir is on Drive, checkpoints from a previous (possibly
    # disconnected) session are still here.  If any exist, resume from the
    # latest one instead of starting over.
    resume = False
    if os.path.exists(output_dir) and any(
        "checkpoint" in d for d in os.listdir(output_dir)
    ):
        print("Found existing checkpoints on Drive! Resuming from the latest...")
        resume = True
    else:
        print("No checkpoints found. Starting fresh!")

    # ---- 6. Train ----
    model_label = "CPT" if is_cpt else "BASE"
    print(f"\nStarting lemmatization fine-tuning ({model_label} model)...")
    trainer.train(resume_from_checkpoint=resume)

    # ---- 7. Save ----
    print(f"\nTraining complete! Saving adapter to {final_adapter} ...")
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    print(f"Saved to {final_adapter}")

    # ---- 8. Summary ----
    print("\n" + "=" * 64)
    print(f"EXPERIMENT 3 ({'CPT' if is_cpt else 'base'} arm) COMPLETE")
    print(f"Model: Gemma-3-270M {'+ CPT' if is_cpt else '(base)'}")
    print(f"Task: Urdu lemmatization")
    print(f"Adapter saved to: {final_adapter}")
    if micro_test:
        print("\nThis was a MICRO-TEST. Check the loss values above.")
        print("If train loss is decreasing: pipeline is working correctly.")
        print("Re-run without --micro-test for the full training.")
    else:
        print("\nNext step: run the evaluation script to compute accuracy")
        print("on the UDU test split and compare against Hafeez et al. (96.3%).")
    print("=" * 64)


if __name__ == "__main__":
    main()
