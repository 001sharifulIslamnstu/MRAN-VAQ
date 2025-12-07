from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingConfig:
    # dataset / paths
    dataset: str
    train_json: Optional[str] = None
    val_json: Optional[str] = None
    image_root: Optional[str] = None
    answer_vocab: Optional[str] = None
    output_dir: str = "outputs"

    # training
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    max_grad_norm: float = 1.0

    # model / encoders
    device: str = "cuda"
    image_encoder_name: str = "google/vit-base-patch16-224-in21k"
    text_encoder_name: str = "bert-base-uncased"
    text_encoder_name_multilingual: str = "bert-base-multilingual-cased"
    hidden_dim: int = 768
    max_question_len: int = 32
    num_answers: int = 3000  # will be updated once vocab is loaded

    # MRAN-VQA specific
    recursion_depth: int = 4
    use_recursive_attention: bool = True
    use_hierarchical_fusion: bool = True
    use_ags_loss: bool = True
    gamma_ags: float = 0.3  # weight for AGS loss

    # misc
    seed: int = 42
    print_every: int = 50

    def to_dict(self):
        return asdict(self)


def get_default_config(dataset: str) -> TrainingConfig:
    dataset = dataset.lower()
    if dataset == "vqa_v2":
        cfg = TrainingConfig(dataset="vqa_v2", epochs=40)
    elif dataset == "clevr":
        cfg = TrainingConfig(dataset="clevr", epochs=20)
    elif dataset == "bangla_vqa":
        cfg = TrainingConfig(dataset="bangla_vqa", epochs=30)
        cfg.text_encoder_name = cfg.text_encoder_name_multilingual
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return cfg
