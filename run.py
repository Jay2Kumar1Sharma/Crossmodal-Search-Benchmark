"""
run.py  --  single entrypoint for the whole pipeline
====================================================
    python run.py --config configs/default.yaml

Stages:
    1. load + split data
    2. encode images & captions (CLIP)  -> cache embeddings
    3. build vector index
    4. baseline benchmark (stage-1 only) with bootstrap CIs
    5. reranked benchmark (stage-2) on a subsample + paired significance test
    6. write results/metrics.csv, results/summary.json, qualitative PNG

Designed to run on Kaggle/Colab (GPU). The logic-only pieces are unit-tested in
tests/ so this orchestrator is just wiring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import RunConfig
from data import DataModule
from encoders import CLIPEncoder
from indexes import make_index
import evaluation as E
from pipeline import TwoStageRetriever
from rerankers import make_reranker


def _rows_from_aggregate(direction: str, agg: dict) -> dict:
    row = {"direction": direction}
    for metric, v in agg.items():
        row[metric] = v["mean"]
        row[f"{metric}_ci"] = f"[{v['lo']}, {v['hi']}]"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = RunConfig.from_yaml(args.config) if os.path.isfile(args.config) else RunConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("[config]", cfg.to_dict())

    # 1. data ---------------------------------------------------------------
    dm = DataModule(cfg.data_dir).load(max_images=cfg.smoke_n if cfg.smoke_test else None)
    dm.stats()
    n_test = cfg.smoke_n // 2 if cfg.smoke_test else cfg.n_test
    _, test_ids = dm.split_by_image(n_test=n_test, seed=cfg.seed)
    image_ids, image_path_list, caption_texts, caption_imgidx = dm.flatten(test_ids)
    print(f"[data] test images={len(image_ids)} captions={len(caption_texts)}")

    # 2. encode -------------------------------------------------------------
    enc = CLIPEncoder(cfg.model_key)
    from PIL import Image
    t0 = time.time()
    img_emb = enc.encode_images([Image.open(p) for p in image_path_list], cfg.img_batch)
    txt_emb = enc.encode_texts(caption_texts, cfg.txt_batch)
    print(f"[encode] {img_emb.shape} images + {txt_emb.shape} captions in {time.time()-t0:.1f}s")
    np.save(f"{cfg.out_dir}/img_emb.npy", img_emb)
    np.save(f"{cfg.out_dir}/txt_emb.npy", txt_emb)
    with open(f"{cfg.out_dir}/ids.json", "w") as f:
        json.dump({"image_ids": image_ids,
                   "image_path_list": image_path_list,
                   "caption_texts": caption_texts,
                   "caption_imgidx": caption_imgidx}, f)

    # 3-4. baseline benchmark ----------------------------------------------
    base = E.run_full_benchmark(img_emb, txt_emb, caption_imgidx,
                                index_kind=cfg.index_kind, ks=cfg.ks, seed=cfg.seed)
    rows = [_rows_from_aggregate("Text->Image (base)", base["text_to_image"]["aggregate"]),
            _rows_from_aggregate("Image->Text (base)", base["image_to_text"]["aggregate"])]

    # 5. reranked benchmark -------------------------------------------------
    summary = {"config": cfg.to_dict(), "baseline": {
        "text_to_image": base["text_to_image"]["aggregate"],
        "image_to_text": base["image_to_text"]["aggregate"]}}

    if cfg.reranker != "identity":
        rk_kwargs = dict(image_paths=image_path_list, caption_texts=caption_texts)
        if cfg.reranker == "clip_rescore":
            rk_kwargs["encoder"] = CLIPEncoder("max")  # rescore with a stronger CLIP
        reranker = make_reranker(cfg.reranker, **rk_kwargs)

        # Text->Image reranked
        t2i_targets = E._targets_text_to_image(caption_imgidx)
        rr_t2i = E.evaluate_reranked(
            base["text_to_image"]["ranked"], t2i_targets,
            rerank_fn=lambda qi, cand, _t=caption_texts:
                reranker.rerank_text_to_image(_t[qi], cand),
            depth=cfg.rerank_depth, subsample=cfg.rerank_subsample,
            with_map=False, seed=cfg.seed)

        # Image->Text reranked
        i2t_targets = E._targets_image_to_text(caption_imgidx, img_emb.shape[0])
        rr_i2t = E.evaluate_reranked(
            base["image_to_text"]["ranked"], i2t_targets,
            rerank_fn=lambda qi, cand, _p=image_path_list:
                reranker.rerank_image_to_text(Image.open(_p[qi]), cand),
            depth=cfg.rerank_depth, subsample=cfg.rerank_subsample,
            with_map=True, seed=cfg.seed)

        rows += [_rows_from_aggregate(f"Text->Image ({reranker.name})", rr_t2i["aggregate"]),
                 _rows_from_aggregate(f"Image->Text ({reranker.name})", rr_i2t["aggregate"])]

        # significance: did rerank improve R@1 on the SAME queries?
        for tag, base_pq, rr in [("text_to_image", base["text_to_image"]["per_query"], rr_t2i),
                                 ("image_to_text", base["image_to_text"]["per_query"], rr_i2t)]:
            before = base_pq["R@1"][rr["selected"]]
            delta = E.paired_bootstrap_delta(before, rr["per_query"]["R@1"], seed=cfg.seed)
            summary.setdefault("rerank_significance", {})[tag] = delta
            print(f"[rerank] {tag} R@1 delta {delta['delta']:+.2f} "
                  f"[{delta['lo']}, {delta['hi']}] significant={delta['significant']}")

    # 6. write artifacts ----------------------------------------------------
    df = pd.DataFrame(rows).set_index("direction")
    df.to_csv(f"{cfg.out_dir}/metrics.csv")
    with open(f"{cfg.out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + df.to_string())
    print(f"\nsaved -> {cfg.out_dir}/metrics.csv, summary.json, *.npy")


if __name__ == "__main__":
    main()
