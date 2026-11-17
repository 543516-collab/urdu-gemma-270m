"""
diagnose_speed_cce.py
======================
Same forward/backward timing test, but with Unsloth's loaded model and
an explicit attempt to confirm/force Cut Cross-Entropy (CCE) usage,
since cut_cross_entropy is already installed as a dependency but may not
be auto-applying to Gemma-3-270M.
"""
import os
# Some Unsloth versions gate optimizations behind env vars -- try the
# common ones before loading anything.
os.environ.setdefault("UNSLOTH_ENABLE_CCE", "1")

import time
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_from_disk

MODEL_NAME = "unsloth/gemma-3-270m"
TOKENIZED_DATASET_PATH = "./tokenized_packed_2048_270m"
MAX_SEQ_LENGTH = 2048

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)

# Inspect whether cut_cross_entropy is importable and whether Unsloth
# references it anywhere on this model instance.
try:
    import cut_cross_entropy
    print(f"\ncut_cross_entropy is importable. Version info: {cut_cross_entropy.__file__}")
except ImportError as e:
    print(f"\ncut_cross_entropy NOT importable: {e}")

# Check if the loaded model's forward function mentions CCE/cce anywhere
import inspect
try:
    src = inspect.getsource(model.forward)
    if "cce" in src.lower() or "cut_cross" in src.lower():
        print("Model's forward() appears to reference CCE.")
    else:
        print("Model's forward() does NOT appear to reference CCE.")
except Exception as e:
    print(f"Could not inspect model.forward source: {e}")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=3407,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)

print("\nLoading dataset...")
dataset = load_from_disk(TOKENIZED_DATASET_PATH)

model.train()

batch_size = 8
indices = list(range(batch_size))
input_ids = torch.stack([
    torch.tensor(dataset[i]["input_ids"]) for i in indices
]).to("cuda")
labels = input_ids.clone()

print(f"\nBatch shape: {input_ids.shape}")
print(f"GPU memory before warmup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

print("\nWarmup pass...")
torch.cuda.synchronize()
t0 = time.time()
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
loss.backward()
model.zero_grad()
torch.cuda.synchronize()
print(f"Warmup pass took: {time.time()-t0:.2f}s")
print(f"Logits returned: {'None (CCE likely active)' if outputs.logits is None else outputs.logits.shape}")

N = 10
print(f"\nTiming {N} forward+backward passes...")
fwd_times, bwd_times = [], []
for i in range(N):
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    torch.cuda.synchronize()
    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.time()
    model.zero_grad()
    fwd_times.append(t1 - t0)
    bwd_times.append(t2 - t1)
    print(f"  step {i}: forward={t1-t0:.3f}s backward={t2-t1:.3f}s loss={loss.item():.3f}")

avg_fwd = sum(fwd_times) / N
avg_bwd = sum(bwd_times) / N
print(f"\nAverage forward time:  {avg_fwd:.3f}s")
print(f"Average backward time: {avg_bwd:.3f}s")
print(f"Average total: {avg_fwd+avg_bwd:.3f}s")
print(f"Compare to original Unsloth measurement: 3.245s")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
