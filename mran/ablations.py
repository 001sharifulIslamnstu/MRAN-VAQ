from typing import Optional

import torch

from .config import TrainingConfig
from .train_eval import create_dataloaders, load_answer_vocab, evaluate_epoch
from .model import MRANVQAModel


def _eval_variant(
    cfg: TrainingConfig,
    variant_name: str,
    use_ra: bool,
    use_hf: bool,
    use_ags: bool,
    ckpt_path: Optional[str],
):
    device = cfg.device
    ans2id, id2ans = load_answer_vocab(cfg.answer_vocab)
    cfg.num_answers = len(ans2id)

    _, val_loader = create_dataloaders(cfg)

    cfg.use_recursive_attention = use_ra
    cfg.use_hierarchical_fusion = use_hf
    cfg.use_ags_loss = use_ags

    model = MRANVQAModel(cfg, num_answers=cfg.num_answers).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Ablation:{variant_name}] Loaded checkpoint from {ckpt_path}")

    metrics = evaluate_epoch(model, val_loader, cfg, id2ans, device)
    print(
        f"[Ablation:{variant_name}] "
        f"acc={metrics['accuracy']:.4f} "
        f"BLEU={metrics['BLEU']:.4f} "
        f"METEOR={metrics['METEOR']:.4f}"
    )
    return metrics


def run_ablations(cfg: TrainingConfig, ckpt_path: Optional[str] = None) -> None:
    """
    Evaluate:
      - Full MRAN-VQA
      - w/o Recursive Attention
      - w/o Hierarchical Fusion
      - w/o AGS loss
    """
    print("\n[Ablation] Full MRAN-VQA")
    _eval_variant(cfg, "full", use_ra=True, use_hf=True, use_ags=True, ckpt_path=ckpt_path)

    print("\n[Ablation] w/o Recursive Attention")
    _eval_variant(cfg, "no_ra", use_ra=False, use_hf=True, use_ags=True, ckpt_path=ckpt_path)

    print("\n[Ablation] w/o Hierarchical Fusion")
    _eval_variant(cfg, "no_hf", use_ra=True, use_hf=False, use_ags=True, ckpt_path=ckpt_path)

    print("\n[Ablation] w/o AGS loss")
    _eval_variant(cfg, "no_ags", use_ra=True, use_hf=True, use_ags=False, ckpt_path=ckpt_path)
