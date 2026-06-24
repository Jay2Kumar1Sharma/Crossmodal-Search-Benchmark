"""
main.py  (FastAPI service, v2)
==============================
Serves the two-stage retriever. Stage-2 rerank is a per-request toggle.

    POST /search/text-to-image   {"query": "...", "k": 5, "rerank": true}
    POST /search/image-to-text   multipart file=@img.jpg ?k=5&rerank=true

Run:
    export EMB_DIR=results/embeddings IMAGES_DIR=data/flickr8k/Images \
           MODEL_KEY=best RERANKER=blip_itm
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from encoders import CLIPEncoder           # noqa: E402
from indexes import make_index             # noqa: E402
from pipeline import TwoStageRetriever      # noqa: E402
from rerankers import make_reranker         # noqa: E402

EMB_DIR = os.environ.get("EMB_DIR", "results/embeddings")
IMAGES_DIR = os.environ.get("IMAGES_DIR", "data/flickr8k/Images")
MODEL_KEY = os.environ.get("MODEL_KEY", "best")
RERANKER = os.environ.get("RERANKER", "identity")

app = FastAPI(title="CLIP Multi-Modal Retrieval (two-stage)", version="2.0")
_R: TwoStageRetriever | None = None


def get_retriever() -> TwoStageRetriever:
    global _R
    if _R is None:
        meta = json.load(open(os.path.join(EMB_DIR, "ids.json")))
        image_path_list = meta["image_path_list"]
        caption_texts = meta["caption_texts"]
        img_emb = np.load(os.path.join(EMB_DIR, "img_emb.npy"))
        txt_emb = np.load(os.path.join(EMB_DIR, "txt_emb.npy"))
        enc = CLIPEncoder(MODEL_KEY)
        reranker = None
        if RERANKER != "identity":
            kw = dict(image_paths=image_path_list, caption_texts=caption_texts)
            if RERANKER == "clip_rescore":
                kw["encoder"] = CLIPEncoder("max")
            reranker = make_reranker(RERANKER, **kw)
        _R = TwoStageRetriever(
            enc,
            make_index("flat_ip").build(img_emb),
            make_index("flat_ip").build(txt_emb),
            image_path_list, caption_texts, reranker)
    return _R


class TextQuery(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_KEY, "reranker": RERANKER}


@app.post("/search/text-to-image")
def t2i(q: TextQuery):
    r = get_retriever().search_text_to_image(q.query, k=q.k, rerank=q.rerank)
    return {"query": q.query,
            "results": [{"path": p, "score": s} for p, s in zip(r.payloads, r.scores)]}


@app.post("/search/image-to-text")
async def i2t(file: UploadFile = File(...), k: int = 5, rerank: bool = True):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = get_retriever().search_image_to_text(img, k=k, rerank=rerank)
    return {"filename": file.filename,
            "results": [{"caption": c, "score": s} for c, s in zip(r.payloads, r.scores)]}
