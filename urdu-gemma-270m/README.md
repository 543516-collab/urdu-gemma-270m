# Urdu continued pre-training of Gemma-3-270M: POS tagging and lemmatization

Repository: https://github.com/543516-collab/urdu-gemma-270m

Code, data splits and results for the Master's thesis
**"Exploiting large language models for improvements in Urdu NLP"**
(Muhammad Talha Sial, Faculty of Informatics, Masaryk University, 2026;
advisor RNDr. Miloš Jakubíček, Ph.D.).

The thesis continues the pre-training of the compact
[Gemma-3-270M](https://huggingface.co/google/gemma-3-270m) model on a
~1.45-billion-token web-crawled Urdu corpus using LoRA, then fine-tunes the base
and the adapted model for POS tagging and lemmatization on the Universal
Dependencies Urdu treebank (UD_Urdu-UDTB).

**Main findings.** Continued pre-training reduces perplexity on held-out Urdu by
48.3% (115.35 → 59.58), but gives no measurable downstream gain: its visible
effect is on output-format stability. The compact model reaches 95.43% lemma
accuracy (unaligned tokens counted as errors) and 85.36 POS Macro-F1; UDPipe 2
remains stronger on this treebank.

## Repository layout

```
cpt/                         continued pre-training
  tokenize_and_pack.py       tokenize the corpus, pack into 2048-token blocks
  train_cpt.py               CPT with LoRA (the reported 23,378-step run)
  evaluate_perplexity.py     base vs CPT perplexity on the UDU dev split
  make_training_figure.py    Figure 4.1 from trainer_state.json
  diagnostics/               throughput profiling used during development
finetune/                    downstream experiments
  finetune_lemma.py          Experiment 3: lemmatization (base / CPT arm)
  finetune_task.py           Experiment 1 (joint) and Experiment 2 (POS)
  evaluate_lemma.py          strict positional evaluation, Experiment 3
  evaluate_task.py           strict positional evaluation, Experiments 1-2
  rescore_joint.py           word-aligned rescoring, Experiment 1
  rescore_pos_f1.py          word-aligned per-tag F1 / Macro-F1, Experiment 2
  rescore_all.py             reproduces every word-aligned / full figure
data/                        UD_Urdu-UDTB splits as prompt/completion JSONL
results/
  predictions/               model outputs on the 535 test sentences
  cpt/                       CPT trainer_state.json and the loss figure
  finetune_logs/             trainer_state.json of each fine-tuning run
```

## Trained adapters

The LoRA adapters are published separately on Hugging Face:

| Adapter | Description |
|---|---|
| `urdu-gemma-270m-cpt` | Continued pre-training adapter (Chapter 4) |
| `urdu-gemma-270m-lemma-base`, `-lemma-cpt` | Experiment 3, base / CPT arm |
| `urdu-gemma-270m-pos-base`, `-pos-cpt` | Experiment 2, base / CPT arm |
| `urdu-gemma-270m-joint-cpt` | Experiment 1, joint POS + lemma |

<!-- Replace with the Hugging Face links after uploading. -->

## Setup

Versions used for the reported results: PyTorch 2.10.0 (CUDA 12.8),
Transformers 5.5.0, Unsloth 2026.6.8, PEFT 0.19.1.

```
pip install -r requirements.txt
```

Gemma models require accepting Google's licence on Hugging Face; set `HF_TOKEN`
if needed.

## Reproducing the results

### 1. Continued pre-training (NVIDIA L4, ~80 h)

The web-crawled corpus itself is not distributed (see "Data" below).  Given a
folder of cleaned JSONL documents (`{"text": ...}` per line):

```
cd cpt
python tokenize_and_pack.py --input "corpus/*.jsonl" --output tokenized_packed_2048_270m
python train_cpt.py --data tokenized_packed_2048_270m --micro-test   # 200-step check
python train_cpt.py --data tokenized_packed_2048_270m                # full epoch
```

Settings: LoRA r=16, α=16 on all seven linear projections, 16-bit LoRA in bf16,
effective batch 32, AdamW 8-bit, peak learning rate 2×10⁻⁵ with 500 warm-up
steps and linear decay, one epoch (23,378 steps).  Re-running the script resumes
from the latest checkpoint.

### 2. Perplexity (intrinsic evaluation)

```
python evaluate_perplexity.py base --dev ur_udtb-ud-dev.conllu
python evaluate_perplexity.py cpt  --dev ur_udtb-ud-dev.conllu --adapter urdu-gemma-270m-lora-final
```

### 3. Fine-tuning and evaluation

The fine-tuning scripts read and write everything in one working folder, set
with `URDU_DIR`.  Copy the data files and the CPT adapter there:

```
mkdir work && cp data/*.jsonl work/
cp -r <cpt-adapter-folder> work/urdu-gemma-270m-lora-final
export URDU_DIR=$PWD/work
cd finetune

python finetune_lemma.py --model base && python evaluate_lemma.py --model base   # Exp. 3
python finetune_lemma.py --model cpt  && python evaluate_lemma.py --model cpt
python finetune_task.py --task pos --model base && python evaluate_task.py --task pos --model base   # Exp. 2
python finetune_task.py --task pos --model cpt  && python evaluate_task.py --task pos --model cpt
python finetune_task.py --task joint --model cpt && python evaluate_task.py --task joint --model cpt # Exp. 1
python rescore_joint.py
python rescore_pos_f1.py --file $URDU_DIR/eval_pos_cpt_predictions.jsonl
```

Settings: a fresh LoRA (r=16, α=16) per task, learning rate 2×10⁻⁴, batch 8,
5 epochs (2,530 steps), sequence length 512, dev loss evaluated every 100 steps,
and the checkpoint with the lowest dev loss kept.  For the CPT arm the CPT
adapter is merged into the base weights first.  Decoding is greedy.

### 4. Checking the reported scores

The released predictions can be rescored without a GPU:

```
cd finetune
python rescore_all.py --data ../data --pred ../results/predictions
```

This prints the word-aligned and full-denominator figures for every arm.  Note
on alignment: Experiments 1 and 2 score every aligned word pair, whereas the
Experiment 3 word-aligned figures in the thesis (96.25 / 96.34) count only pairs
with identical surface words; `rescore_all.py` reports both rules.

## Data

- **UD_Urdu-UDTB** (data as of UD v2.14; 4,043 / 552 / 535 sentences).  The files
  in `data/` and the predictions in `results/predictions/` are derived from it and
  are therefore released under **CC BY-NC-SA 4.0**; see `data/README.md`.
- **Web-crawled pre-training corpus** (~1.45 B tokens, built with Spiderling,
  JusText, Onion and Unitok as described in Chapter 3).  It is not distributed
  here because it consists of third-party web text.

## Licences

- Code: MIT (see `LICENSE`).
- `data/` and `results/predictions/`: CC BY-NC-SA 4.0, following UD_Urdu-UDTB.
- Adapters: subject to the Gemma Terms of Use; the task adapters are also
  trained on CC BY-NC-SA 4.0 data.

## Citation

```
@mastersthesis{sial2026urdu,
  author = {Sial, Muhammad Talha},
  title  = {Exploiting large language models for improvements in Urdu NLP},
  school = {Masaryk University, Faculty of Informatics},
  year   = {2026}
}
```
