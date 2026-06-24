"""
indexes.py
==========
Concrete VectorIndex implementations behind a single interface, so the pipeline
and evaluation code never care whether search is exact or approximate.

  FlatIPIndex : exact inner-product (cosine on normalized vectors). Optimal recall,
                the right choice at this scale (a few thousand vectors).
  IVFFlatIndex: inverted-file ANN. Trades a little recall for speed at large scale.
  HNSWIndex   : graph-based ANN. Strong speed/recall trade-off, no training.

The point of shipping all three is to *demonstrate* the exact-vs-approximate
trade-off (see results/ann_tradeoff.png from the notebook), then justify using
Flat for the actual benchmark because the gallery is small.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import faiss


class _Base:
    name = "base"

    def __init__(self):
        self.index = None

    def _prep(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype="float32")

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.ntotal)
        return self.index.search(self._prep(queries), k)

    @property
    def ntotal(self) -> int:
        return 0 if self.index is None else self.index.ntotal


class FlatIPIndex(_Base):
    name = "flat_ip"

    def build(self, embeddings: np.ndarray) -> "FlatIPIndex":
        emb = self._prep(embeddings)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return self


class IVFFlatIndex(_Base):
    name = "ivf_flat"

    def __init__(self, nlist: int = 100, nprobe: int = 16):
        super().__init__()
        self.nlist, self.nprobe = nlist, nprobe

    def build(self, embeddings: np.ndarray) -> "IVFFlatIndex":
        from utils import suppress_stderr
        emb = self._prep(embeddings)
        d = emb.shape[1]
        # FAISS wants >= ~39*nlist training points; clamp nlist so it never
        # under-trains (which would emit a C-level clustering warning).
        nlist = max(1, min(self.nlist, emb.shape[0] // 39, emb.shape[0]))
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        with suppress_stderr():
            self.index.train(emb)
            self.index.add(emb)
        self.index.nprobe = min(self.nprobe, nlist)
        return self


class HNSWIndex(_Base):
    name = "hnsw"

    def __init__(self, M: int = 32, ef_search: int = 64):
        super().__init__()
        self.M, self.ef_search = M, ef_search

    def build(self, embeddings: np.ndarray) -> "HNSWIndex":
        emb = self._prep(embeddings)
        self.index = faiss.IndexHNSWFlat(emb.shape[1], self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(emb)
        return self


def make_index(kind: str = "flat_ip", **kwargs):
    return {"flat_ip": FlatIPIndex,
            "ivf_flat": IVFFlatIndex,
            "hnsw": HNSWIndex}[kind](**kwargs)
