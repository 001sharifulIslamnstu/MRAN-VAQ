from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from .model import MRANVQAModel
from .config import TrainingConfig


def _load_checkpoint(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = TrainingConfig(**cfg_dict)
    ans2id = ckpt["ans2id"]
    id2ans = {int(k): v for k, v in ckpt["id2ans"].items()}
    cfg.num_answers = len(ans2id)

    model = MRANVQAModel(cfg, num_answers=cfg.num_answers).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, cfg, ans2id, id2ans


def run_qualitative_examples(
    model_ckpt: str,
    image_paths: List[str],
    questions: List[str],
    id2ans: Optional[dict] = None,
    device_str: str = "cuda",
):
    """
    Simple function to print predictions for custom image-question pairs.
    """

    model, cfg, ans2id, ckpt_id2ans = _load_checkpoint(model_ckpt, device=device_str)
    if id2ans is None:
        id2ans = ckpt_id2ans

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_name)

    img_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    assert len(image_paths) == len(questions), "image_paths and questions must match in length"

    for img_path, question in zip(image_paths, questions):
        img = Image.open(img_path).convert("RGB")
        img_t = img_transform(img).unsqueeze(0).to(device_str)

        tokens = tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=cfg.max_question_len,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device_str)
        attention_mask = tokens["attention_mask"].to(device_str)

        with torch.no_grad():
            logits, attn = model(img_t, input_ids, attention_mask)
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_ans = id2ans.get(pred_id, "")

        print(f"\n[Qualitative]")
        print(f"Image: {img_path}")
        print(f"Q: {question}")
        print(f"Predicted: {pred_ans}")
