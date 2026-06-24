"""
metrics.py
==========
IR metrics computed as PER-QUERY arrays first, then aggregated with bootstrap
confidence intervals. Reporting a 95% CI (e.g. R@1 = 61.4 [59.8, 63.1]) instead
of a bare point estimate is the kind of rigor that stands out in an interview --
it says "I know my eval set is finite and my number has uncertainty."

Relevance is binary. Two query types are supported via `target_sets`:
  * single-target (text->image): each query has exactly one correct gallery item
  * multi-target (image->text):  each query has several correct gallery items

All functions take a FULL ranking (gallery indices, best-first) per query so that
MRR / mAP / nDCG see ranks beyond the recall cut-offs.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Per-query primitives
# --------------------------------------------------------------------------- #
def first_hit_ranks(ranked_idx: np.ndarray, target_sets: Sequence[set]) -> np.ndarray:
    """1-indexed rank of the first relevant item per query (len+1 if none)."""
    ranks = np.full(len(ranked_idx), ranked_idx.shape[1] + 1, dtype=np.int64)
    for i, row in enumerate(ranked_idx):
        tg = target_sets[i]
        for r, j in enumerate(row, 1):
            if j in tg:
                ranks[i] = r
                break
    return ranks


def average_precision_row(row: np.ndarray, targets: set) -> float:
    if not targets:
        return 0.0
    hits, precs = 0, []
    for r, j in enumerate(row, 1):
        if j in targets:
            hits += 1
            precs.append(hits / r)
    return float(np.sum(precs) / len(targets)) if precs else 0.0


def ndcg_row(row: np.ndarray, targets: set, k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not targets:
        return 0.0
    dcg = 0.0
    for pos, j in enumerate(row[:k], 1):
        if j in targets:
            dcg += 1.0 / np.log2(pos + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(targets), k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


# --------------------------------------------------------------------------- #
# Per-query metric bundle
# --------------------------------------------------------------------------- #
def per_query_metrics(ranked_idx: np.ndarray,
                      target_sets: Sequence[set],
                      ks: Sequence[int] = (1, 5, 10),
                      with_map: bool = True,
                      ndcg_k: int = 10) -> Dict[str, np.ndarray]:
    """Return a dict of per-query arrays (length = n_queries)."""
    ranks = first_hit_ranks(ranked_idx, target_sets)
    out: Dict[str, np.ndarray] = {f"R@{k}": (ranks <= k).astype(np.float64) for k in ks}
    out["MRR"] = 1.0 / ranks
    out[f"nDCG@{ndcg_k}"] = np.array(
        [ndcg_row(ranked_idx[i], target_sets[i], ndcg_k) for i in range(len(ranked_idx))])
    if with_map:
        out["mAP"] = np.array(
            [average_precision_row(ranked_idx[i], target_sets[i]) for i in range(len(ranked_idx))])
    return out


# --------------------------------------------------------------------------- #
# Aggregation with bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap CI for the mean of a per-query metric array."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


def aggregate(per_query: Dict[str, np.ndarray], as_percent: bool = True,
              n_boot: int = 1000, seed: int = 0) -> Dict[str, dict]:
    """
    Collapse per-query arrays into {metric: {mean, lo, hi}} with bootstrap CIs.
    Recall/mAP/nDCG are scaled to percent; MRR stays in [0, 1].
    """
    result = {}
    for name, arr in per_query.items():
        mean, lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
        scale = 100.0 if (as_percent and name != "MRR") else 1.0
        result[name] = {"mean": round(mean * scale, 2),
                        "lo": round(lo * scale, 2),
                        "hi": round(hi * scale, 2)}
    return result


def format_row(direction: str, agg: Dict[str, dict]) -> str:
    parts = [f"{direction:<14}"]
    for name, v in agg.items():
        parts.append(f"{name}={v['mean']:.2f} [{v['lo']:.2f},{v['hi']:.2f}]")
    return "  ".join(parts)
