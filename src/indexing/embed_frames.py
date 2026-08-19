"""
embed_frames.py

Wraps open_clip to turn video frames (or text queries) into embedding vectors.

IMPORTANT: torch / open_clip are imported lazily inside ClipEmbedder.__init__,
NOT at module level. This means other modules (build_index.py, search.py,
frame_sampler.py, the FastAPI app) can import FrameEmbedding and type-hint
against ClipEmbedder without paying the cost of loading torch, and without
requiring torch to even be installed unless embedding actually happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class FrameEmbedding:
    """Embedding vector for a single sampled frame, plus provenance metadata."""

    video_path: str
    frame_index: int
    timestamp_sec: float
    vector: np.ndarray = field(repr=False)  # shape (embed_dim,), L2-normalized float32

    def to_metadata_dict(self) -> dict:
        """Everything except the raw vector — what gets persisted to metadata.json."""
        return {
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
        }


class ClipEmbedder:
    """
    Thin wrapper around open_clip for embedding frames (images) and text queries
    into the same shared embedding space.

    Model defaults to ViT-B-32 / openai pretrained weights, matching requirements.txt.
    """

    # NOTE: openai's ViT-B-32 checkpoint was trained with QuickGELU activation.
    # Using the plain "ViT-B-32" arch name silently mismatches (quick_gelu=False)
    # against the "openai" pretrained tag (quick_gelu=True), which degrades
    # embedding quality without erroring. "ViT-B-32-quickgelu" fixes this.
    MODEL_NAME = "ViT-B-32-quickgelu"
    PRETRAINED = "openai"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device: Optional[str] = None,
    ):
        # Lazy imports: only touched when someone actually constructs a ClipEmbedder.
        import torch
        import open_clip

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.embed_dim = self.model.visual.output_dim

    # ---------- image / frame embedding ----------

    def embed_frame(self, bgr_image: np.ndarray) -> np.ndarray:
        """Embed a single frame (as a BGR numpy array from cv2) -> (embed_dim,) float32, L2-normalized."""
        return self.embed_frames_batch([bgr_image])[0]

    def embed_frames_batch(self, bgr_images: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR numpy frames -> (N, embed_dim) float32, L2-normalized rows."""
        import cv2
        from PIL import Image

        torch = self.torch
        tensors = []
        for bgr in bgr_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil_img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def embed_sampled_frames(self, sampled_frames) -> List[FrameEmbedding]:
        """
        Convenience: takes an iterable of SampledFrame (from frame_sampler.py)
        and returns a list of FrameEmbedding, batched internally for speed.
        """
        sampled_frames = list(sampled_frames)
        if not sampled_frames:
            return []

        images = [f.image for f in sampled_frames]
        vectors = self.embed_frames_batch(images)

        return [
            FrameEmbedding(
                video_path=f.video_path,
                frame_index=f.frame_index,
                timestamp_sec=f.timestamp_sec,
                vector=vec,
            )
            for f, vec in zip(sampled_frames, vectors)
        ]

    # ---------- text embedding ----------

    def embed_text(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Embed one or more text strings -> (embed_dim,) if a single string was passed,
        or (N, embed_dim) if a list was passed. Always L2-normalized.
        """
        torch = self.torch
        single = isinstance(text, str)
        texts = [text] if single else list(text)

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        vectors = features.cpu().numpy().astype(np.float32)
        return vectors[0] if single else vectors


if __name__ == "__main__":
    # Smoke test — only runs if torch/open_clip are actually installed.
    import sys

    embedder = ClipEmbedder()
    print(f"Loaded {embedder.MODEL_NAME} ({embedder.PRETRAINED}) on {embedder.device}, "
          f"embed_dim={embedder.embed_dim}")

    text_vec = embedder.embed_text("a red car driving down the street")
    print(f"Text embedding shape: {text_vec.shape}, norm={np.linalg.norm(text_vec):.4f}")