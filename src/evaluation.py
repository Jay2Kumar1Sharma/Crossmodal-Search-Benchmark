"""
evaluation.py
=============
Benchmark harness. Produces, for both retrieval directions:
  * Recall@1/5/10, MRR, mAP, nDCG@10 with 95% bootstrap confidence intervals
  * an optional reranked variant (stage-2) evaluated on a query subsample
  * a paired bootstrap significance test (does reranking actually help?)

It operates on precomputed embeddings so stage-1 is cheap; the (expensive)
reranker is only invoked on the subsample used for the reranked numbers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from indexes import make_index
from metrics import aggregate, per_query_metrics
from pipeline import splice_reranked


def _targets_text_to_image(caption_imgidx: Sequence[int]) -> List[set]:
    return [{caption_imgidx[i]} for i in range(len(caption_imgidx))]


def _targets_image_to_text(caption_imgidx: Sequence[int], n_images: int) -> List[set]:
    m = defaultdict(set)
    for crow, ipos in enumerate(caption_imgidx):
        m[ipos].add(crow)
    return [m[i] for i in range(n_images)]


def _full_ranking(index, query_emb: np.ndarray) -> np.ndarray:
    _, ranked = index.search(query_emb, index.ntotal)
    return ranked


def evaluate_direction(query_emb: np.ndarray,
                       gallery_emb: np.ndarray,
                       target_sets: List[set],
                       ks=(1, 5, 10),
                       with_map=True,
                       index_kind="flat_ip",
                       n_boot=1000,
                       seed=0) -> Dict:
    """Baseline (stage-1 only) evaluation for one direction."""
    index = make_index(index_kind).build(gallery_emb)
    ranked = _full_ranking(index, query_emb)
    pq = per_query_metrics(ranked, target_sets, ks=ks, with_map=with_map)
    return {"aggregate": aggregate(pq, n_boot=n_boot, seed=seed),
            "per_query": pq, "ranked": ranked, "index": index}


def evaluate_reranked(ranked_stage1: np.ndarray,
                      target_sets: List[set],
                      rerank_fn: Callable[[int, List[int]], List[int]],
                      depth: int = 50,
                      subsample: int = 500,
                      ks=(1, 5, 10),
                      with_map=True,
                      n_boot=1000,
                      seed=0) -> Dict:
    """
    Apply a reranker to the top-`depth` of each (subsampled) query's stage-1
    ranking and recompute metrics.

    rerank_fn(query_row_index, candidate_ids) -> reordered candidate_ids
    """
    rng = np.random.default_rng(seed)
    n = len(ranked_stage1)
    sel = np.arange(n) if subsample >= n else rng.choice(n, size=subsample, replace=False)

    reranked_rows, sub_targets = [], []
    for qi in sel:
        row = ranked_stage1[qi]
        head = row[:depth].tolist()
        reordered = rerank_fn(int(qi), head)
        reranked_rows.append(splice_reranked(row, reordered))
        sub_targets.append(target_sets[qi])
    reranked = np.vstack(reranked_rows)

    pq = per_query_metrics(reranked, sub_targets, ks=ks, with_map=with_map)
    return {"aggregate": aggregate(pq, n_boot=n_boot, seed=seed),
            "per_query": pq, "n_queries": len(sel), "selected": sel}


def paired_bootstrap_delta(before: np.ndarray, after: np.ndarray,
                           n_boot=1000, seed=0) -> Dict:
    """
    Paired bootstrap on the SAME queries: is mean(after) - mean(before) > 0?
    Returns the mean delta and a 95% CI; CI excluding 0 => significant.
    """
    before, after = np.asarray(before), np.asarray(after)
    assert before.shape == after.shape
    rng = np.random.default_rng(seed)
    n = len(before)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = after[idx].mean(1) - before[idx].mean(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float((after - before).mean()) * 100, 2),
            "lo": round(float(lo) * 100, 2), "hi": round(float(hi) * 100, 2),
            "significant": bool(lo > 0 or hi < 0)}


def run_full_benchmark(img_emb, txt_emb, caption_imgidx,
                       index_kind="flat_ip", ks=(1, 5, 10), seed=0) -> Dict:
    """Both directions, baseline only. Returns a tidy results dict."""
    t2i = evaluate_direction(txt_emb, img_emb,
                             _targets_text_to_image(caption_imgidx),
                             ks=ks, with_map=False, index_kind=index_kind, seed=seed)
    i2t = evaluate_direction(img_emb, txt_emb,
                             _targets_image_to_text(caption_imgidx, img_emb.shape[0]),
                             ks=ks, with_map=True, index_kind=index_kind, seed=seed)
    return {"text_to_image": t2i, "image_to_text": i2t}
