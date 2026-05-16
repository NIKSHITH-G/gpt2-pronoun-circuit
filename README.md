# GPT-2 Pronoun Resolution Circuit
### Mechanistic Interpretability — Causal Pronoun Tracking in GPT-2 Small

> **Finding:** A sparse 5-head circuit spanning layers 4–9 is causally responsible for 
> pronoun resolution after "because" clauses. Ablating these 5 heads (3.5% of the model) 
> inverts the model's pronoun preference — logit diff flips from +0.107 to -0.038.

---

## What this project does

This project identifies and characterizes the **attention head circuit** GPT-2 Small uses 
to resolve pronouns in causal sentences like:

> "Alice gave Bob the book because ___ was kind" → **she**  
> "Bob helped Alice move because ___ was strong" → **he**

Using activation patching, head ablation, and attention visualization, I traced this 
behavior to a specific set of 5 heads — and found that the most critical head (L7H3) 
functions as a **subject-tracking head**, consistently attending to the grammatical 
subject and verb.

---

## Key Results

| Finding | Result |
|---|---|
| Baseline accuracy | 75% |
| Baseline logit diff | +0.107 |
| Ablated accuracy (5 heads removed) | 55% |
| Ablated logit diff | -0.038 |
| % of signal removed | 135% |
| Heads used / total heads | 5 / 144 (3.5%) |

The logit diff going **negative** means the ablated model now prefers the *wrong* pronoun — 
the circuit isn't just contributing to the behavior, it's what makes it possible.

---

## Circuit — Top 5 Heads

| Rank | Layer | Head | Patching Score | Role |
|------|-------|------|---------------|------|
| 1 | 7 | 1 | -0.0999 | Late subject resolution |
| 2 | 4 | 11 | -0.0984 | Early entity encoding |
| 3 | 9 | 5 | -0.0734 | Final layer refinement |
| 4 | 7 | 3 | -0.0612 | **Subject + verb tracking** |
| 5 | 8 | 3 | -0.0262 | Mid-layer integration |

**L7H3** consistently attends to the grammatical subject and verb across all prompts, 
suggesting it identifies the *agent* of the action rather than just tracking names.

---

## Repo Structure
gpt2-pronoun-circuit/
│
├── pronoun_circuit_analysis.py   # Full analysis code (copy into Colab)
├── alignment_forum_post.md       # Write-up of findings
└── README.md

---

## How to Run

**Requirements:** Google Colab with T4 GPU (free tier works fine)

**Step 1 — Install libraries**
```bash
pip install transformer_lens circuitsvis plotly einops jaxtyping
```

**Step 2 — Open `pronoun_circuit_analysis.py` in Colab**

Each `# ── CELL N` comment marks a new notebook cell. Run them in order.

**Step 3 — Expected runtime**

~15 minutes total on a T4 GPU. The patching loop (Cell 9) is the slowest — 
it runs 144 patching experiments per prompt.

---

## Methods Overview

### Logit Difference Metric
Instead of argmax accuracy, I measure the gap between log-probabilities for the 
correct vs incorrect pronoun. This gives a continuous signal that's more sensitive 
to partial circuit damage.

### Activation Patching
For each prompt, a corrupted version is created by swapping subject/object names 
(Alice↔Bob, Sarah↔John, Emma↔James). Activations are patched head-by-head from 
the clean run into the corrupted run. Heads that most restore the correct logit 
difference are causally important.

### Ablation
Zero-ablation: setting a head's value vectors to 0, removing its contribution 
entirely. Used to confirm causal importance and test circuit redundancy.

---

## Limitations

- **Small dataset** (20 prompts) — results should be validated at larger scale
- **Name-gender confound** — GPT-2 associates Alice with female, Bob with male; 
  some correct predictions may come from name lookup rather than syntactic reasoning
- **Non-causal comparison** only has 2 prompts — not enough to draw conclusions
- Attention patterns are suggestive but not causal evidence on their own

---

## Related Work

- Wang et al. (2022) — [Interpretability in the Wild: IOI Circuit in GPT-2](https://arxiv.org/abs/2211.00593)
- Elhage et al. (2021) — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Conmy et al. (2023) — [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997)
- Nanda et al. (2023) — [200 Concrete Open Problems in Mechanistic Interpretability](https://www.lesswrong.com/posts/LbrPTJ4fmABEdEnLf/)

---
