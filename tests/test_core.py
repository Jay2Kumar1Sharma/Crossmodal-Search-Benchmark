"""
Test suite for the retrieval system's logic-heavy components.
Run: pytest -q   (from repo root, with src/ on PYTHONPATH)

These use SYNTHETIC embeddings and a MOCK reranker so the metric/index/pipeline
logic is verified with zero GPU and zero model downloads.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metrics as M
from indexes import make_index
from pipeline import splice_reranked
import evaluation as E


# ---------- fixtures ----------
def perfect_setup(n_img=6, caps=5, D=32, noise=0.0, seed=0):
    """Each image a distinct vector; its captions = that vector (+noise)."""
    rng = np.random.default_rng(seed)
    img = np.eye(n_img, D, dtype="float32")
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = np.repeat(img, caps, axis=0) + noise * rng.standard_normal((n_img * caps, D)).astype("float32")
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    caption_imgidx = [i // caps for i in range(n_img * caps)]
    return img, txt, caption_imgidx


# ---------- metrics ----------
def test_metrics_perfect_retrieval():
    img, txt, cidx = perfect_setup()
    res = E.run_full_benchmark(img, txt, cidx)
    t2i = res["text_to_image"]["aggregate"]
    i2t = res["image_to_text"]["aggregate"]
    assert t2i["R@1"]["mean"] == 100.0
    assert t2i["MRR"]["mean"] == 1.0
    assert i2t["R@1"]["mean"] == 100.0
    assert i2t["mAP"]["mean"] == 100.0
    assert i2t["nDCG@10"]["mean"] == 100.0


def test_average_precision_known_value():
    # gallery of 4, relevant {0,2}, ranking puts relevant at ranks 2 and 4:
    row = np.array([1, 0, 3, 2])
    ap = M.average_precision_row(row, {0, 2})
    # AP = (1/2)*(prec@2 + prec@4) = (1/2)*(1/2 + 2/4) = 0.5
    assert abs(ap - 0.5) < 1e-9


def test_ndcg_monotonic_in_rank():
    targets = {0}
    good = M.ndcg_row(np.array([0, 1, 2, 3]), targets, 10)  # relevant at rank 1
    bad = M.ndcg_row(np.array([1, 2, 3, 0]), targets, 10)   # relevant at rank 4
    assert good == 1.0 and 0 < bad < 1.0 and good > bad


def test_bootstrap_ci_brackets_mean():
    vals = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    mean, lo, hi = M.bootstrap_ci(vals, n_boot=2000, seed=1)
    assert abs(mean - 0.6) < 1e-9 and lo <= mean <= hi


# ---------- indexes ----------
@pytest.mark.parametrize("kind", ["flat_ip", "ivf_flat", "hnsw"])
def test_index_finds_self(kind):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 16)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = make_index(kind).build(emb)
    scores, ids = idx.search(emb[:20], k=1)
    # top-1 for a vector should (overwhelmingly) be itself; allow ANN slack.
    self_hits = sum(ids[i, 0] == i for i in range(20))
    assert self_hits >= 18, f"{kind} only matched {self_hits}/20 to themselves"


def test_ann_approximates_exact():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((500, 32)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[:50]
    flat = make_index("flat_ip").build(emb).search(q, 10)[1]
    hnsw = make_index("hnsw").build(emb).search(q, 10)[1]
    # recall@10 of HNSW vs exact neighbours should be high
    overlap = np.mean([len(set(flat[i]) & set(hnsw[i])) / 10 for i in range(50)])
    assert overlap > 0.8


# ---------- pipeline splice ----------
def test_splice_reranked_keeps_full_ranking():
    full = np.array([5, 4, 3, 2, 1, 0])
    reordered_head = [3, 5, 4]          # rerank of the first 3 ids {5,4,3}
    out = splice_reranked(full, reordered_head)
    assert out.tolist() == [3, 5, 4, 2, 1, 0]
    assert sorted(out.tolist()) == sorted(full.tolist())  # no items lost/added


# ---------- reranked evaluation + significance ----------
def test_reranking_improves_and_is_detected():
    img, txt, cidx = perfect_setup(n_img=40, caps=5, D=64, noise=0.4, seed=3)
    base = E.evaluate_direction(txt, img, E._targets_text_to_image(cidx), with_map=False)
    ranked_stage1 = base["ranked"]
    targets = E._targets_text_to_image(cidx)

    # Oracle reranker: pull the true target to the front if it's in the head.
    def oracle(qi, cand_ids):
        tgt = next(iter(targets[qi]))
        return ([tgt] + [c for c in cand_ids if c != tgt]) if tgt in cand_ids else list(cand_ids)

    rr = E.evaluate_reranked(ranked_stage1, targets, oracle, depth=20,
                             subsample=200, with_map=False, seed=3)
    # paired test on R@1 over the SAME subsampled queries
    sel = rr["selected"]
    before = base["per_query"]["R@1"][sel]
    after = rr["per_query"]["R@1"]
    delta = E.paired_bootstrap_delta(before, after, seed=3)
    assert rr["aggregate"]["R@1"]["mean"] >= base["aggregate"]["R@1"]["mean"]
    assert delta["delta"] >= 0


# ---------- LoRA contrastive loss (numpy reference of the torch formula) ----------
def _contrastive_loss_np(img, txt, scale):
    """Mirror of finetune.clip_contrastive_loss in numpy, to validate the math."""
    logits = scale * img @ txt.T            # (N, N)
    n = logits.shape[0]

    def ce(lg):
        lg = lg - lg.max(axis=1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        return -np.mean(logp[np.arange(n), np.arange(n)])

    return 0.5 * (ce(logits) + ce(logits.T))


def test_contrastive_loss_rewards_alignment():
    rng = np.random.default_rng(0)
    D, N = 32, 16
    base = rng.standard_normal((N, D)).astype("float32")
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # aligned: text == image (diagonal is the right answer)
    aligned = _contrastive_loss_np(base, base.copy(), scale=10.0)
    # misaligned: shuffle text so the diagonal is usually wrong
    perm = rng.permutation(N)
    misaligned = _contrastive_loss_np(base, base[perm], scale=10.0)
    assert aligned < misaligned, (aligned, misaligned)
    # perfectly aligned + high temperature -> loss approaches 0
    assert _contrastive_loss_np(base, base.copy(), scale=100.0) < 0.01
