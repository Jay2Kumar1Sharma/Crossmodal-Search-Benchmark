"""
rerankers.py
============
Stage-2 rerankers implementing the Reranker interface. Given a query and the
stage-1 candidate ids, return them reordered best-first.

  IdentityReranker      : no-op. The honest ablation baseline ("rerank off").
  CLIPRescoreReranker   : rescore the head with a (typically larger) CLIP. Cheap,
                          no extra model family. "retrieve small, rerank large".
  BLIPReranker          : true cross-encoder. A BLIP image-text-matching (ITM)
                          head attends over image AND text jointly, which a
                          bi-encoder cannot -- this is the strongest reranker and
                          the architectural centerpiece.

Rerankers hold references to the gallery payloads (image paths aligned to the
image gallery, caption texts aligned to the text gallery) so they can fetch the
actual content for a candidate id.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
class IdentityReranker:
    name = "identity"

    def rerank_text_to_image(self, query_text, candidate_ids): return list(candidate_ids)
    def rerank_image_to_text(self, query_image, candidate_ids): return list(candidate_ids)


# --------------------------------------------------------------------------- #
class CLIPRescoreReranker:
    """
    Recompute cosine similarity for the head using a (possibly different/larger)
    CLIP encoder, then reorder. If the rescoring encoder == retrieval encoder the
    order is unchanged, so pass a stronger model_key here (e.g. 'max').
    """
    name = "clip_rescore"

    def __init__(self, encoder, image_paths: Sequence[str], caption_texts: Sequence[str]):
        self.enc = encoder
        self.image_paths = list(image_paths)
        self.caption_texts = list(caption_texts)

    def rerank_text_to_image(self, query_text: str, candidate_ids: Sequence[int]) -> List[int]:
        if not candidate_ids:
            return list(candidate_ids)
        q = self.enc.encode_texts([query_text], batch_size=1)            # (1, D)
        imgs = [Image.open(self.image_paths[i]).convert("RGB") for i in candidate_ids]
        cand = self.enc.encode_images(imgs)                              # (C, D)
        sims = (cand @ q[0])
        order = np.argsort(-sims)
        return [candidate_ids[i] for i in order]

    def rerank_image_to_text(self, query_image: Image.Image, candidate_ids: Sequence[int]) -> List[int]:
        if not candidate_ids:
            return list(candidate_ids)
        q = self.enc.encode_images([query_image])                       # (1, D)
        cand = self.enc.encode_texts([self.caption_texts[i] for i in candidate_ids])
        sims = (cand @ q[0])
        order = np.argsort(-sims)
        return [candidate_ids[i] for i in order]


# --------------------------------------------------------------------------- #
class BLIPReranker:
    """
    Cross-encoder reranker using BLIP's image-text-matching (ITM) head
    (`Salesforce/blip-itm-base-coco`). For each (image, text) candidate pair it
    returns P(match); we sort candidates by that probability.
    """
    name = "blip_itm"

    def __init__(self, image_paths: Sequence[str], caption_texts: Sequence[str],
                 model_name: str = "Salesforce/blip-itm-base-coco",
                 device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        from utils import suppress_stderr
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with suppress_stderr():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device).eval()
        self.image_paths = list(image_paths)
        self.caption_texts = list(caption_texts)
        self.bs = batch_size

    def _itm_scores(self, images, texts) -> np.ndarray:
        """P(match) for paired lists images[i] <-> texts[i]."""
        torch = self.torch
        scores = []
        for s in range(0, len(images), self.bs):
            imgs = images[s:s + self.bs]
            txts = texts[s:s + self.bs]
            inp = self.processor(images=imgs, text=txts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model(**inp, use_itm_head=True)
                prob = torch.softmax(out.itm_score, dim=1)[:, 1]  # col 1 = "matched"
            scores.append(prob.float().cpu().numpy())
        return np.concatenate(scores, 0)

    def rerank_text_to_image(self, query_text: str, candidate_ids: Sequence[int]) -> List[int]:
        if not candidate_ids:
            return list(candidate_ids)
        imgs = [Image.open(self.image_paths[i]).convert("RGB") for i in candidate_ids]
        scores = self._itm_scores(imgs, [query_text] * len(candidate_ids))
        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]

    def rerank_image_to_text(self, query_image: Image.Image, candidate_ids: Sequence[int]) -> List[int]:
        if not candidate_ids:
            return list(candidate_ids)
        txts = [self.caption_texts[i] for i in candidate_ids]
        scores = self._itm_scores([query_image.convert("RGB")] * len(candidate_ids), txts)
        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]


def make_reranker(kind: str, **kwargs):
    return {"identity": IdentityReranker,
            "clip_rescore": CLIPRescoreReranker,
            "blip_itm": BLIPReranker}[kind](**kwargs)
