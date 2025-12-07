from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .datasets import VQADataset
from .collate import MRANCollator
from .model import MRANVQAModel
from .train_eval import load_answer_vocab, evaluate_epoch, create_dataloaders


def run_recursion_sweep(cfg: TrainingConfig, ckpt_path: Optional[str] = None) -> None:
    """
    Sweep recursion depth R = 1..6 and print accuracy + latency.

    If ckpt_path is provided, the same trained weights are used and only
    recursion depth is changed.
    """
    device = cfg.device
    ans2id, id2ans = load_answer_vocab(cfg.answer_vocab)
    cfg.num_answers = len(ans2id)

    _, val_loader = create_dataloaders(cfg)

    model = MRANVQAModel(cfg, num_answers=cfg.num_answers).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Sweep] Loaded checkpoint from {ckpt_path}")

    import time

    results = []
    for R in [1, 2, 3, 4, 5, 6]:
        print(f"\n[SWEEP] Evaluating recursion depth R={R}")
        model.cfg.recursion_depth = R

        # measure latency with dummy input
        B = 1
        dummy_img = torch.randn(B, 3, 224, 224, device=device)
        dummy_input_ids = torch.ones(
            B, cfg.max_question_len, dtype=torch.long, device=device
        )
        dummy_attn = torch.ones(
            B, cfg.max_question_len, dtype=torch.long, device=device
        )

        model.eval()
        with torch.no_grad():
            # warm-up
            for _ in range(5):
                _ = model(dummy_img, dummy_input_ids, dummy_attn, recursion_depth=R)

            iters = 20
            t0 = time.time()
            for _ in range(iters):
                _ = model(dummy_img, dummy_input_ids, dummy_attn, recursion_depth=R)
            t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0 / iters

        # full evaluation on val set
        metrics = evaluate_epoch(model, val_loader, cfg, id2ans, device)
        print(
            f"[SWEEP][R={R}] "
            f"acc={metrics['accuracy']:.4f} "
            f"BLEU={metrics['BLEU']:.4f} "
            f"METEOR={metrics['METEOR']:.4f} "
            f"latency_ms={latency_ms:.2f}"
        )

        results.append(
            {
                "R": R,
                "metrics": metrics,
                "latency_ms": latency_ms,
            }
        )

    print("\n[SWEEP] Summary:")
    for r in results:
        print(
            f"R={r['R']}: acc={r['metrics']['accuracy']:.4f}, "
            f"BLEU={r['metrics']['BLEU']:.4f}, "
            f"METEOR={r['metrics']['METEOR']:.4f}, "
            f"latency_ms={r['latency_ms']:.2f}"
        )
