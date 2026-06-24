# Results

- `metrics_baseline.csv` — zero-shot stage-1 CLIP (LAION ViT-L/14) on the Flickr8k
  1,000-image held-out test split, with 95% bootstrap confidence intervals.

Regenerate everything (and add reranked / LoRA numbers + the qualitative grid) by
running `notebooks/clip_retrieval_kaggle.ipynb` on Kaggle (GPU T4). The notebook
also writes `qualitative_examples.png` and `summary.json` to `/kaggle/working`.
