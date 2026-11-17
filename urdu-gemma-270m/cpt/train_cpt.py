"""
train_cpt.py
============
Continued pre-training (CPT) of Gemma-3-270M on the packed Urdu corpus with
LoRA (Chapter 4 of the thesis).  This is the configuration of the reported
run: single epoch, 23,378 optimisation steps on one NVIDIA L4 (24 GB),
about 80 hours.

    LoRA r=16, alpha=16, dropout 0, all 7 linear projections
    16-bit LoRA (no quantization), bf16, no gradient checkpointing
    per-device batch 8 x grad. accumulation 4 = effective batch 32
    AdamW 8-bit, peak LR 2e-5, 500 warm-up steps, linear decay
    weight decay 0.01, max grad norm 1.0, sequence length 2048

Checkpoints are written every 500 steps; re-running the script resumes from
the latest checkpoint in --output-dir (the reported run was resumed twice).

Usage:
    python train_cpt.py --data tokenized_packed_2048_270m              # full run
    python train_cpt.py --data tokenized_packed_2048_270m --micro-test # 200 steps
"""
import argparse
import os
import time

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tokenized_packed_2048_270m")
    ap.add_argument("--output-dir", default="urdu_gemma_270m_l4_checkpoints")
    ap.add_argument("--final-dir", default="urdu-gemma-270m-l4-lora-final")
    ap.add_argument("--model", default="unsloth/gemma-3-270m")
    ap.add_argument("--micro-test", action="store_true",
                    help="200-step sanity run before the full epoch")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=None,
        load_in_4bit=False)

    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing=False, random_state=3407,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True)

    dataset = load_from_disk(args.data)
    print(dataset)
    print("Sample:", tokenizer.decode(dataset[0]["input_ids"][:60],
                                      skip_special_tokens=True))

    common = dict(per_device_train_batch_size=8, gradient_accumulation_steps=4,
                  learning_rate=2e-5, max_grad_norm=1.0,
                  bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
                  optim="adamw_8bit", weight_decay=0.01,
                  output_dir=args.output_dir, logging_steps=10,
                  dataloader_num_workers=2, report_to="none")
    if args.micro_test:
        training_args = TrainingArguments(warmup_steps=20, max_steps=200,
                                          save_steps=200, save_total_limit=1, **common)
    else:
        training_args = TrainingArguments(warmup_steps=500, num_train_epochs=1,
                                          save_steps=500, save_total_limit=3, **common)

    trainer = Trainer(
        model=model, train_dataset=dataset, args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer,
            mlm=False))

    resume = os.path.isdir(args.output_dir) and any(
        "checkpoint" in d for d in os.listdir(args.output_dir))
    print("Resuming from latest checkpoint" if resume else "Starting fresh")

    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print(f"Elapsed: {(time.time() - start) / 3600:.2f} h")

    model.save_pretrained(args.final_dir)
    tokenizer.save_pretrained(args.final_dir)
    print("Saved adapter to", args.final_dir)


if __name__ == "__main__":
    main()
