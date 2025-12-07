# MRAN-VQA: Multimodal Recursive Attention Network for Visual Question Answering

This repository contains a PyTorch implementation of **MRAN-VQA** and the **Attention Grounding Score (AGS)** following:

> MRAN-VQA: Multimodal Recursive Attention Network for Visual Question Answering  
> *Engineering Science and Technology, an International Journal (JESTECH), 2025, Vol. 72, Article 102232*  
> DOI: **10.1016/j.jestch.2025.102232**

The code supports:

- Training **MRAN-VQA** on **VQA v2.0**, **CLEVR**, and **BanglaVQA**
- Computing **VQA accuracy**, **BLEU**, **METEOR**, and **AGS+**
- **Recursive attention depth sweep** (R = 1…6) with latency measurement
- **Ablation studies** (w/o Recursive Attention, w/o Hierarchical Fusion, w/o AGS loss)
- **Model statistics** (parameter count, single-sample latency)
- Simple scripts for **qualitative examples** and **AGS analysis**

---

## 1. Repository Structure

```text
.
├── main.py                       # Entry point: train / sweep_R / ablation / stats
├── mran/
│   ├── __init__.py
│   ├── config.py                 # TrainingConfig dataclass
│   ├── datasets.py               # Generic VQA-style dataset loader
│   ├── collate.py                # Collator: image transforms + tokenization
│   ├── ags.py                    # Attention Grounding Score (AGS) + loss
│   ├── model.py                  # MRAN-VQA model + recursive attention block
│   ├── metrics.py                # VQA soft accuracy, BLEU, METEOR, helpers
│   ├── train_eval.py             # Training loop + evaluation
│   ├── recursion_sweep.py        # R-sweep experiments (accuracy + latency)
│   ├── ablations.py              # Ablation study utilities
│   ├── model_stats.py            # Parameter count + latency measurement
│   └── qualitative.py            # Qualitative examples / visualization helper
└── README.md                     # This file



