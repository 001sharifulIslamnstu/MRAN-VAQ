from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, AutoModel

from .config import TrainingConfig


class RecursiveAttentionBlock(nn.Module):
    """
    Simple recursive cross-attention block:
    - Given text state (B, H) and image features (B, P, H),
      iteratively refines the text state using attention over image patches.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_img = nn.Linear(hidden_dim, hidden_dim)
        self.W_txt = nn.Linear(hidden_dim, hidden_dim)
        self.W_attn = nn.Linear(hidden_dim, 1)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, txt_state: Tensor, img_feats: Tensor, R: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            txt_state: (B, H)
            img_feats: (B, P, H)
            R: recursion depth

        Returns:
            txt_state: (B, H) refined
            attn: (B, P) final attention map
        """
        B, P, H = img_feats.shape
        attn = None

        for _ in range(R):
            img_proj = self.W_img(img_feats)  # (B, P, H)
            txt_proj = self.W_txt(txt_state).unsqueeze(1)  # (B, 1, H)
            h = torch.tanh(img_proj + txt_proj)  # (B, P, H)
            attn_logits = self.W_attn(h).squeeze(-1)  # (B, P)
            attn = F.softmax(attn_logits, dim=-1)  # (B, P)

            img_context = torch.bmm(attn.unsqueeze(1), img_feats).squeeze(1)  # (B, H)
            fused = torch.cat([txt_state, img_context], dim=-1)  # (B, 2H)
            txt_state = torch.tanh(self.fuse(fused))  # (B, H)

        return txt_state, attn


class MRANVQAModel(nn.Module):
    """
    MRAN-VQA model with:
      - ViT image encoder
      - BERT/MBERT text encoder
      - Recursive visual attention over image patches
      - Simple hierarchical fusion & classifier head
    """

    def __init__(self, cfg: TrainingConfig, num_answers: int):
        super().__init__()
        self.cfg = cfg
        self.num_answers = num_answers

        # Encoders
        self.image_encoder = ViTModel.from_pretrained(cfg.image_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_encoder_name)

        hidden_dim = cfg.hidden_dim

        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, hidden_dim)
        self.txt_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # Recursive attention block
        self.rec_block = RecursiveAttentionBlock(hidden_dim)

        # Hierarchical fusion: combine (txt_state, img_global)
        self.fuse_hier = nn.Linear(2 * hidden_dim, hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        recursion_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            pixel_values: (B, 3, H, W)
            input_ids: (B, L)
            attention_mask: (B, L)
            recursion_depth: overrides cfg.recursion_depth if provided

        Returns:
            logits: (B, num_answers)
            attn: (B, P) visual attention over patches
        """
        if recursion_depth is None:
            recursion_depth = self.cfg.recursion_depth

        # Image encoding
        img_out = self.image_encoder(pixel_values=pixel_values)
        img_feats = img_out.last_hidden_state  # (B, P+1, D_vit)
        img_feats = img_feats[:, 1:, :]  # drop CLS token → (B, P, D_vit)
        img_feats = self.img_proj(img_feats)  # (B, P, H)

        # Text encoding
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_out.last_hidden_state[:, 0, :]  # CLS token: (B, D_bert)
        txt_state = self.txt_proj(txt_cls)  # (B, H)

        # Recursive attention
        if self.cfg.use_recursive_attention:
            txt_state, attn = self.rec_block(txt_state, img_feats, R=recursion_depth)
        else:
            # no recursion: simple attention once
            txt_state, attn = self.rec_block(txt_state, img_feats, R=1)

        # Hierarchical fusion (here: fuse txt_state with global pooled image)
        img_global = img_feats.mean(dim=1)  # (B, H)
        if self.cfg.use_hierarchical_fusion:
            fused = torch.cat([txt_state, img_global], dim=-1)
            fused = torch.tanh(self.fuse_hier(fused))
        else:
            fused = txt_state

        logits = self.classifier(fused)  # (B, num_answers)
        return logits, attn
