"""
gradio_app.py
=============
Interactive demo for the two-stage retriever -- far more presentable in an
interview than curl. Two tabs: text->image and image->text, each with a
"rerank" toggle so you can SHOW the stage-2 effect live.

Run (after embeddings exist in EMB_DIR):
    python app/gradio_app.py
On Kaggle/Colab call build_demo(retriever).launch(share=True) from the notebook.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build_demo(retriever):
    import gradio as gr
    from PIL import Image

    def t2i(query, rerank, k):
        res = retriever.search_text_to_image(query, k=int(k), rerank=rerank)
        return [(Image.open(p), f"score={s:.3f}" if s == s else "reranked")
                for p, s in zip(res.payloads, res.scores)]

    def i2t(image, rerank, k):
        res = retriever.search_image_to_text(image, k=int(k), rerank=rerank)
        return "\n".join(f"{i+1}. {c}" for i, c in enumerate(res.payloads))

    with gr.Blocks(title="CLIP + FAISS + BLIP retrieval") as demo:
        gr.Markdown("# Multi-modal retrieval — CLIP bi-encoder + FAISS + BLIP rerank")
        with gr.Tab("Text → Image"):
            q = gr.Textbox(label="Text query", value="a dog running on the beach")
            with gr.Row():
                rr = gr.Checkbox(label="Stage-2 rerank", value=True)
                k = gr.Slider(1, 10, value=5, step=1, label="top-K")
            gallery = gr.Gallery(label="Results", columns=5, height=240)
            gr.Button("Search").click(t2i, [q, rr, k], gallery)
        with gr.Tab("Image → Text"):
            img = gr.Image(type="pil", label="Query image")
            with gr.Row():
                rr2 = gr.Checkbox(label="Stage-2 rerank", value=True)
                k2 = gr.Slider(1, 10, value=5, step=1, label="top-K")
            out = gr.Textbox(label="Retrieved captions", lines=6)
            gr.Button("Search").click(i2t, [img, rr2, k2], out)
    return demo


if __name__ == "__main__":
    print("Import build_demo(retriever) from the notebook after building a "
          "TwoStageRetriever, or wire up embeddings here first.")
