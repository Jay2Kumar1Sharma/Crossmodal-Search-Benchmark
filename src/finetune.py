"""
finetune.py
===========
Parameter-efficient fine-tuning of CLIP for retrieval, using LoRA adapters on the
attention projections of BOTH the vision and text towers (PEFT). Only the adapters
train (<1% of parameters); the base weights stay frozen.

Objective = CLIP's own symmetric contrastive (InfoNCE) loss over a batch of N
distinct (image, caption) pairs: the N x N similarity matrix should be largest on
its diagonal. We sample ONE caption per image per step so a batch never contains
two captions of the same image (which would create false negatives).

Usage (on GPU):
    enc = CLIPEncoder("fast")                      # baseline encoder
    # ... evaluate baseline ...
    enc = train_lora(enc, train_ids, image_paths, image_to_captions, epochs=2)
    # enc now produces LoRA-adapted embeddings; re-embed + re-evaluate.

The contrastive-loss math is unit-tested (numpy reference) in tests/.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Loss (torch imported lazily so the module loads without torch present)
# --------------------------------------------------------------------------- #
def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """
    Symmetric InfoNCE. image_embeds/text_embeds are L2-normalized (N, D);
    logit_scale is a scalar tensor. Returns a scalar loss.
    """
    import torch
    import torch.nn.functional as F
    logits = logit_scale * image_embeds @ text_embeds.t()      # (N, N)
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_pairs(train_ids: Sequence[str],
                image_paths: Dict[str, str],
                image_to_captions: Dict[str, List[str]]):
    """List of (image_path, [captions]) for the training images."""
    return [(image_paths[i], image_to_captions[i]) for i in train_ids]


def _make_dataset(pairs, processor, seed=0):
    import torch
    from PIL import Image

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.pairs = pairs
            self.rng = random.Random(seed)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            path, caps = self.pairs[idx]
            return Image.open(path).convert("RGB"), self.rng.choice(caps)

    def collate(batch):
        imgs, caps = zip(*batch)
        pix = processor(images=list(imgs), return_tensors="pt")["pixel_values"]
        tok = processor(text=list(caps), return_tensors="pt", padding=True, truncation=True)
        return pix, tok["input_ids"], tok["attention_mask"]

    return PairDataset(), collate


# --------------------------------------------------------------------------- #
# LoRA wiring + training
# --------------------------------------------------------------------------- #
def apply_lora(clip_model, r=8, alpha=16, dropout=0.05,
               targets=("q_proj", "k_proj", "v_proj", "out_proj")):
    """Inject LoRA adapters into both encoders' attention projections."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=list(targets), bias="none")
    peft_model = get_peft_model(clip_model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def train_lora(encoder,
               train_ids: Sequence[str],
               image_paths: Dict[str, str],
               image_to_captions: Dict[str, List[str]],
               epochs: int = 2,
               lr: float = 1e-4,
               batch_size: int = 128,
               r: int = 8,
               seed: int = 0,
               save_dir: str | None = None):
    """
    Fine-tune `encoder` in place with LoRA. After training, encoder.encode_images/
    encode_texts produce adapted embeddings. Returns the same encoder.
    """
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    device = encoder.device
    peft_model = apply_lora(encoder.model, r=r)
    clip = peft_model.base_model.model           # underlying CLIPModel (adapters injected)
    peft_model.to(device).train()

    pairs = build_pairs(train_ids, image_paths, image_to_captions)
    ds, collate = _make_dataset(pairs, encoder.processor, seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=collate, num_workers=2, drop_last=True)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    if len(loader) == 0:
        print("  (training set too small for this batch_size — skipping LoRA training)")
        peft_model.eval()
        encoder.model = clip
        return encoder

    for ep in range(epochs):
        running, n = 0.0, 0
        for pix, ids, mask in tqdm(loader, desc=f"LoRA epoch {ep+1}/{epochs}"):
            pix, ids, mask = pix.to(device), ids.to(device), mask.to(device)
            with torch.autocast(device):
                _vout = clip.vision_model(pixel_values=pix)
                _tout = clip.text_model(input_ids=ids, attention_mask=mask)
                _img = clip.visual_projection(_vout.pooler_output)
                _txt = clip.text_projection(_tout.pooler_output)
                img = F.normalize(_img, dim=-1)
                txt = F.normalize(_txt, dim=-1)
                scale = clip.logit_scale.exp().clamp(max=100.0)
                loss = clip_contrastive_loss(img, txt, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        print(f"  epoch {ep+1}: mean contrastive loss = {running/max(n,1):.4f}")

    peft_model.eval()
    if save_dir:
        peft_model.save_pretrained(save_dir)
        print(f"  saved LoRA adapter -> {save_dir}")
    # Route the encoder's feature calls through the adapted CLIPModel.
    encoder.model = clip
    return encoder
