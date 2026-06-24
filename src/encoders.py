"""
encoders.py
===========
Concrete Encoder built on a Hugging Face CLIP model. Implements the Encoder
interface so the rest of the system never references CLIP directly.

Model menu (only the string changes -- all load via transformers.CLIPModel):
    fast : openai/clip-vit-base-patch32        (baseline)
    b16  : openai/clip-vit-base-patch16
    best : laion/CLIP-ViT-L-14-laion2B-s32B-b82K   (recommended)
    max  : laion/CLIP-ViT-H-14-laion2B-s32B-b79K
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils import suppress_stderr

MODELS = {
    "fast": "openai/clip-vit-base-patch32",
    "b16":  "openai/clip-vit-base-patch16",
    "best": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "max":  "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class CLIPEncoder:
    """Encoder Protocol implementation."""

    def __init__(self, model_key: str = "best", device: str | None = None):
        self.name = MODELS.get(model_key, model_key)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        with suppress_stderr():
            self.model = CLIPModel.from_pretrained(self.name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(self.name)
        self.dim = self.model.config.projection_dim
        print(f"[CLIPEncoder] {self.name} on {self.device} | dim={self.dim} "
              f"| load {time.time()-t0:.1f}s")

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            inp = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.autocast(self.device):
                # Explicit projection path — stable across transformers versions
                # (get_image_features' return type changed in newer releases).
                vout = self.model.vision_model(pixel_values=inp["pixel_values"])
                feat = self.model.visual_projection(vout.pooler_output)
            out.append(feat.float().cpu().numpy())
        return _l2(np.concatenate(out, 0))

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            inp = self.processor(text=list(texts[i:i + batch_size]), return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            with torch.autocast(self.device):
                tout = self.model.text_model(input_ids=inp["input_ids"],
                                             attention_mask=inp["attention_mask"])
                feat = self.model.text_projection(tout.pooler_output)
            out.append(feat.float().cpu().numpy())
        return _l2(np.concatenate(out, 0))
