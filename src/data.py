"""
data.py
=======
Flickr8k data module: parse captions, keep images that exist, split BY IMAGE
(so no caption leaks across train/test), and flatten the test split into the
parallel arrays the pipeline consumes.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DataModule:
    data_dir: str
    images_sub: str = "Images"
    captions_file: str = "captions.txt"
    image_to_captions: Dict[str, List[str]] = field(default_factory=dict)
    image_paths: Dict[str, str] = field(default_factory=dict)

    def load(self, max_images: int | None = None) -> "DataModule":
        images_dir = os.path.join(self.data_dir, self.images_sub)
        caps_path = os.path.join(self.data_dir, self.captions_file)
        assert os.path.isdir(images_dir), f"No Images/ at {images_dir}"
        assert os.path.isfile(caps_path), f"No captions at {caps_path}"

        with open(caps_path, encoding="utf-8") as f:
            lines = f.readlines()
        start = 1 if lines and lines[0].lower().startswith("image") else 0
        raw: Dict[str, List[str]] = {}
        for ln in lines[start:]:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            parts = ln.split(",", 1)          # first comma only
            if len(parts) == 2:
                raw.setdefault(parts[0].strip(), []).append(parts[1].strip())

        for img, caps in raw.items():
            p = os.path.join(images_dir, img)
            if os.path.isfile(p):
                self.image_to_captions[img] = caps
                self.image_paths[img] = os.path.abspath(p)

        if max_images:
            keep = sorted(self.image_to_captions)[:max_images]
            self.image_to_captions = {k: self.image_to_captions[k] for k in keep}
            self.image_paths = {k: self.image_paths[k] for k in keep}
        return self

    def stats(self) -> dict:
        lens = [len(c.split()) for caps in self.image_to_captions.values() for c in caps]
        n_img = len(self.image_to_captions)
        s = {"images": n_img, "captions": len(lens),
             "caps_per_image": round(len(lens) / max(n_img, 1), 2),
             "avg_caption_len": round(sum(lens) / max(len(lens), 1), 2)}
        print("[data]", s)
        return s

    def split_by_image(self, n_test: int = 1000, seed: int = 42) -> Tuple[List[str], List[str]]:
        ids = sorted(self.image_to_captions)
        random.Random(seed).shuffle(ids)
        n_test = min(n_test, max(1, len(ids) // 5))
        return sorted(ids[n_test:]), sorted(ids[:n_test])

    def flatten(self, image_ids: List[str]):
        """Return (image_ids, image_path_list, caption_texts, caption_imgidx)."""
        path_list = [self.image_paths[i] for i in image_ids]
        texts, imgidx = [], []
        for pos, img in enumerate(image_ids):
            for cap in self.image_to_captions[img]:
                texts.append(cap)
                imgidx.append(pos)
        return image_ids, path_list, texts, imgidx
