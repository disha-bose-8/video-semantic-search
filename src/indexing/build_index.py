"""
build_index.py

Builds, saves, and loads a FAISS IndexFlatIP over frame embeddings, plus a
parallel metadata.json that maps each FAISS row -> (video_path, frame_index,
timestamp_sec). No torch/open_clip dependency here — only faiss + numpy —
so this module stays cheap to import.

IndexFlatIP + L2-normalized vectors == cosine similarity search.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import List, Optional, Sequence

import faiss
import numpy as np


class ClipIndex:
    """
    Wraps a faiss.IndexFlatIP plus its associated metadata list.

    metadata[i] corresponds to the vector stored at row i of the FAISS index.
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.index = faiss.IndexFlatIP(embed_dim)
        self.metadata: List[dict] = []

    # ---------- building ----------

    def add(self, vectors: np.ndarray, metadata_entries: Sequence[dict]) -> None:
        """
        Add a batch of vectors + their metadata dicts.

        vectors: (N, embed_dim) float32, ideally L2-normalized (cosine via inner product).
        metadata_entries: length-N sequence of dicts, one per vector.
        """
        if len(vectors) != len(metadata_entries):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata_entries ({len(metadata_entries)}) "
                "must be the same length"
            )
        if len(vectors) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got vectors with dim={vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.metadata.extend(metadata_entries)

    def add_frame_embeddings(self, frame_embeddings) -> None:
        """
        Convenience: accepts a list of FrameEmbedding-like objects (from embed_frames.py)
        or plain dicts with a 'vector' key. Extracts vectors + metadata and calls add().
        """
        vectors = []
        metadata_entries = []
        for fe in frame_embeddings:
            if is_dataclass(fe):
                d = asdict(fe)
                vec = d.pop("vector")
                metadata_entries.append(d)
            elif isinstance(fe, dict):
                d = dict(fe)
                vec = d.pop("vector")
                metadata_entries.append(d)
            else:
                # Object with .vector and .to_metadata_dict()
                vec = fe.vector
                metadata_entries.append(fe.to_metadata_dict())
            vectors.append(np.asarray(vec, dtype=np.float32))

        if vectors:
            self.add(np.vstack(vectors), metadata_entries)

    # ---------- searching ----------

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        """
        Search for the top_k nearest neighbors of query_vector (shape (embed_dim,)).

        Returns (scores, metadata_hits):
            scores: list[float] cosine similarity scores, descending
            metadata_hits: list[dict] metadata entries corresponding to each hit
        """
        if self.index.ntotal == 0:
            return [], []

        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)

        scores = scores[0].tolist()
        indices = indices[0].tolist()

        hits = []
        valid_scores = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            hits.append(self.metadata[idx])
            valid_scores.append(score)

        return valid_scores, hits

    # ---------- persistence ----------

    def save(self, index_dir: str) -> None:
        """Saves index.faiss + metadata.json + config.json into index_dir."""
        os.makedirs(index_dir, exist_ok=True)

        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))

        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        with open(os.path.join(index_dir, "config.json"), "w") as f:
            json.dump({"embed_dim": self.embed_dim}, f, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "ClipIndex":
        """Loads an index previously saved with .save()."""
        config_path = os.path.join(index_dir, "config.json")
        index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.json")

        for p in (config_path, index_path, metadata_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing index file: {p}")

        with open(config_path) as f:
            config = json.load(f)

        obj = cls(embed_dim=config["embed_dim"])
        obj.index = faiss.read_index(index_path)

        with open(metadata_path) as f:
            obj.metadata = json.load(f)

        return obj

    def __len__(self) -> int:
        return self.index.ntotal


# ---------- end-to-end helper: build an index directly from a video file ----------

def build_index_from_video(
    video_path: str,
    index_dir: str,
    interval_sec: float = 1.0,
    embedder: Optional[object] = None,
) -> ClipIndex:
    """
    Full pipeline glue: sample frames -> embed -> build FAISS index -> save.

    embedder: pass an already-constructed ClipEmbedder to avoid reloading the
    model repeatedly when indexing many videos. If None, a new one is created
    (this is where the torch/open_clip import actually happens).
    """
    from src.indexing.frame_sampler import FrameSampler
    from src.indexing.embed_frames import ClipEmbedder

    if embedder is None:
        embedder = ClipEmbedder()

    sampler = FrameSampler(interval_sec=interval_sec)
    sampled_frames = list(sampler.sample(video_path))

    if not sampled_frames:
        raise ValueError(f"No frames sampled from {video_path}")

    frame_embeddings = embedder.embed_sampled_frames(sampled_frames)

    clip_index = ClipIndex(embed_dim=embedder.embed_dim)
    clip_index.add_frame_embeddings(frame_embeddings)
    clip_index.save(index_dir)

    return clip_index


if __name__ == "__main__":
    # Mock-data smoke test — no torch/open_clip needed.
    print("Running mock smoke test for ClipIndex...")

    embed_dim = 512
    idx = ClipIndex(embed_dim=embed_dim)

    rng = np.random.default_rng(42)
    n_frames = 20
    vectors = rng.normal(size=(n_frames, embed_dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    metadata_entries = [
        {"video_path": "mock.mp4", "frame_index": i * 30, "timestamp_sec": float(i)}
        for i in range(n_frames)
    ]
    idx.add(vectors, metadata_entries)
    print(f"Index size: {len(idx)}")

    query = vectors[5] + rng.normal(scale=0.01, size=embed_dim).astype(np.float32)
    query /= np.linalg.norm(query)
    scores, hits = idx.search(query, top_k=3)
    print("Top hits:", list(zip(scores, hits)))

    tmp_dir = "/tmp/clip_index_smoke_test"
    idx.save(tmp_dir)
    reloaded = ClipIndex.load(tmp_dir)
    print(f"Reloaded index size: {len(reloaded)} (matches: {len(reloaded) == len(idx)})")
    print("Smoke test passed.")