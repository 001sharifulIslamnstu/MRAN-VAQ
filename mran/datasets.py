import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """
    Generic VQA-style dataset.

    Each JSON entry should have at least:
      - "image": str (relative path or filename)
      - "question": str
      - "answer_id": int (index in answer vocab)
    Optionally:
      - "answers": list of annotation answers (for soft VQA accuracy)
      - "mask": list of floats (AGS mask over patches)
    """

    def __init__(self, json_path: str, image_root: str):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)

        with self.json_path.open("r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = self.image_root / sample["image"]
        image = Image.open(img_path).convert("RGB")

        question = sample["question"]
        answer_id = sample.get("answer_id", -1)
        answers = sample.get("answers", None)
        mask = sample.get("mask", None)
        if mask is not None:
            mask = np.asarray(mask, dtype="float32")

        return {
            "image": image,
            "question": question,
            "answer_id": answer_id,
            "answers": answers,
            "mask": mask,
        }
