"""
interfaces.py
=============
The architectural backbone: small, explicit contracts that let every component
be swapped without touching the rest of the system. This is what turns a script
into a *system* -- you can change the encoder, the index type, or the reranker
independently, and the evaluation/serving code stays identical.

    Encoder      : raw images/text  -> L2-normalized embeddings
    VectorIndex  : embeddings       -> top-k nearest neighbours (exact OR approx)
    Reranker     : (query, candidates) -> reordered candidates (stage-2 scoring)

Everything downstream (pipeline, evaluation harness, API, demo) depends only on
these Protocols, never on a concrete CLIP/FAISS/BLIP class.
"""
from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class Encoder(Protocol):
    """Maps a batch of images or texts into a shared, L2-normalized space."""
    name: str
    dim: int

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray: ...
    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray: ...


@runtime_checkable
class VectorIndex(Protocol):
    """A searchable nearest-neighbour index over a fixed gallery."""
    name: str

    def build(self, embeddings: np.ndarray) -> "VectorIndex": ...
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), each shape (n_queries, k), best-first."""
        ...

    @property
    def ntotal(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    """
    Stage-2 scorer. Given a query and the stage-1 candidate ids, return the
    candidates reordered best-first (optionally with new scores). A reranker may
    use a stronger, cross-modal model than the bi-encoder used for retrieval.
    """
    name: str

    def rerank_text_to_image(self, query_text: str,
                             candidate_ids: Sequence[int]) -> List[int]: ...
    def rerank_image_to_text(self, query_image: Image.Image,
                             candidate_ids: Sequence[int]) -> List[int]: ...
