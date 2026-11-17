#!/usr/bin/env python3
"""Regenerate Figure 4.1 (training loss + learning rate) from trainer_state.json.

Usage:
    python make_training_figure.py path/to/trainer_state.json
Writes figures/training-loss.pdf and figures/training-loss.png and prints the
numbers the thesis text and caption quote, so you can check them against it.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INIT_LOSS = 4.06      # initial loss quoted in the thesis (pre-resume point)
ROLL = 31             # rolling-mean window quoted in the caption

path = sys.argv[1] if len(sys.argv) > 1 else "trainer_state.json"
with open(path) as f:
    state = json.load(f)

rows = [r for r in state["log_history"] if "loss" in r and "step" in r]
rows.sort(key=lambda r: r["step"])
step = np.array([r["step"] for r in rows], dtype=float)
loss = np.array([r["loss"] for r in rows], dtype=float)
lr = np.array([r.get("learning_rate", np.nan) for r in rows], dtype=float)

# Rolling mean over the logged points
k = min(ROLL, len(loss))
roll = np.convolve(loss, np.ones(k) / k, mode="valid")
roll_step = step[k - 1:]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(step, loss, color="#9ecae1", lw=0.6, label="per-step loss")
ax.plot(roll_step, roll, color="#08519c", lw=1.8, label=f"{ROLL}-step rolling mean")

# Dashed connector from the initial value if logging starts late
if step[0] > 50:
    ax.plot([0, step[0]], [INIT_LOSS, loss[0]], color="#08519c", ls="--", lw=1)
    ax.plot([0], [INIT_LOSS], marker="o", color="#08519c", ms=4)

ax.set_xlabel("optimization step")
ax.set_ylabel("training loss")
ax.set_xlim(0, state.get("max_steps", step.max()))
ax.set_ylim(min(2.9, loss.min() - 0.05), max(INIT_LOSS, loss.max()) + 0.1)

ax2 = ax.twinx()
ax2.plot(step, lr, color="grey", lw=1.2, label="learning rate")
ax2.set_ylabel("learning rate")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/training-loss.pdf")
fig.savefig("figures/training-loss.png", dpi=200)

# ---- numbers to check against the thesis text/caption ----
print("max_steps in log        :", state.get("max_steps"), "(thesis: 23378)")
print("last logged step        :", int(step[-1]))
print("first logged step       :", int(step[0]), "(caption says ~2000)")
print("first logged loss       :", round(float(loss[0]), 3), "(thesis: initial ~4.06)")
print("final loss (last 500 st):", round(float(loss[step >= step[-1] - 500].mean()), 3), "(thesis: ~3.05)")
tail = loss[step >= step[-1] - 500]
print("final-stretch range     :", round(float(tail.min()), 3), "-", round(float(tail.max()), 3), "(thesis: 3.02-3.11)")
print("peak learning rate      :", f"{np.nanmax(lr):.3g}", "(thesis: 2e-05)")
print("final learning rate     :", f"{lr[-1]:.3g}", "(comment says ~8e-9)")
print("max loss anywhere       :", round(float(loss.max()), 2), "(should NOT be >10)")
