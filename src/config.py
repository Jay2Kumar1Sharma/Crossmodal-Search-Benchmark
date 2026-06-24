"""
config.py
=========
Single typed configuration object, loadable from YAML. One place for every knob,
which keeps runs reproducible and makes the system self-documenting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    # data
    data_dir: str = "/kaggle/input/flickr8k"
    n_test: int = 1000
    seed: int = 42
    smoke_test: bool = True
    smoke_n: int = 100
    # model / index / rerank
    model_key: str = "best"            # fast | b16 | best | max
    index_kind: str = "flat_ip"        # flat_ip | ivf_flat | hnsw
    reranker: str = "blip_itm"         # identity | clip_rescore | blip_itm
    rerank_depth: int = 50
    rerank_subsample: int = 500        # queries used for the (expensive) reranked eval
    # batching
    img_batch: int = 64
    txt_batch: int = 256
    # eval
    ks: tuple = (1, 5, 10)
    n_boot: int = 1000
    out_dir: str = "/kaggle/working"

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "ks" in data:
            data["ks"] = tuple(data["ks"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
