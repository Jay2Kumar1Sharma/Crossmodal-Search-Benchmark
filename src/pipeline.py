"""
pipeline.py
===========
TwoStageRetriever = bi-encoder retrieval (stage 1) + optional reranking (stage 2).

    stage 1:  encode query -> FAISS top-`depth` candidates           (fast, recall-y)
    stage 2:  reranker re-scores those `depth` candidates            (slow, precise)

This is the standard retrieve-then-rerank pattern used in search and RAG. It is
deliberately model-agnostic: it depends only on the Encoder / VectorIndex /
Reranker interfaces, so you can drop in a bigger CLIP, an ANN index, or a BLIP
cross-encoder without changing this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from interfaces import Encoder, Reranker, VectorIndex


@dataclass
class RetrievalResult:
    ids: List[int]
    scores: List[float]
    payloads: List[str]   # image path (t2i) or caption text (i2t)


class TwoStageRetriever:
    def __init__(self,
                 encoder: Encoder,
                 image_index: VectorIndex,
                 text_index: VectorIndex,
                 image_paths: Sequence[str],
                 caption_texts: Sequence[str],
                 reranker: Optional[Reranker] = None):
        self.encoder = encoder
        self.image_index = image_index
        self.text_index = text_index
        self.image_paths = list(image_paths)      # aligned to image gallery ids
        self.caption_texts = list(caption_texts)  # aligned to text gallery ids
        self.reranker = reranker

    # ---------- serving ----------
    def search_text_to_image(self, text: str, k: int = 5,
                             depth: int = 50, rerank: bool = True) -> RetrievalResult:
        q = self.encoder.encode_texts([text], batch_size=1)
        scores, idx = self.image_index.search(q, max(k, depth if rerank else k))
        cand = idx[0].tolist()
        sc = scores[0].tolist()
        if rerank and self.reranker is not None:
            order = self.reranker.rerank_text_to_image(text, cand[:depth])
            cand = order + cand[depth:]
            sc = [float("nan")] * len(order) + sc[depth:]
        cand, sc = cand[:k], sc[:k]
        return RetrievalResult(cand, sc, [self.image_paths[i] for i in cand])

    def search_image_to_text(self, image: Image.Image, k: int = 5,
                             depth: int = 50, rerank: bool = True) -> RetrievalResult:
        q = self.encoder.encode_images([image], batch_size=1)
        scores, idx = self.text_index.search(q, max(k, depth if rerank else k))
        cand = idx[0].tolist()
        sc = scores[0].tolist()
        if rerank and self.reranker is not None:
            order = self.reranker.rerank_image_to_text(image, cand[:depth])
            cand = order + cand[depth:]
            sc = [float("nan")] * len(order) + sc[depth:]
        cand, sc = cand[:k], sc[:k]
        return RetrievalResult(cand, sc, [self.caption_texts[i] for i in cand])


def splice_reranked(full_row: np.ndarray, reordered_head: Sequence[int]) -> np.ndarray:
    """
    Replace the first len(reordered_head) entries of a full ranking with the
    reranked order, keeping the stage-1 tail. Used by the evaluation harness so
    reranked metrics still have a complete ranking for MRR/mAP/nDCG.
    """
    head_set = set(reordered_head)
    tail = [j for j in full_row if j not in head_set]
    return np.array(list(reordered_head) + tail, dtype=full_row.dtype)
