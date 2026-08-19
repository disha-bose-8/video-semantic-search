"""
scripts/save_flagged_frames.py

Saves the top-K matching frames (from a search) as actual .jpg images you can
open and look at, instead of just reading timestamps off the console.

Usage:
    python scripts/save_flagged_frames.py --video data/raw_videos/YOUR_VIDEO.mp4 --query "a player celebrating a goal" --top_k 10
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2

from src.indexing.frame_sampler import FrameSampler
from src.indexing.embed_frames import ClipEmbedder
from src.indexing.build_index import ClipIndex


def main():
    parser = argparse.ArgumentParser(description="Save top-matching frames as .jpg images")
    parser.add_argument("--video", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_dir", default="data/flagged_frames")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Sampling + embedding...")
    sampler = FrameSampler(interval_sec=args.interval)
    sampled_frames = list(sampler.sample(args.video))

    embedder = ClipEmbedder()
    frame_embeddings = embedder.embed_sampled_frames(sampled_frames)

    clip_index = ClipIndex(embed_dim=embedder.embed_dim)
    clip_index.add_frame_embeddings(frame_embeddings)

    query_vector = embedder.embed_text(args.query)
    scores, hits = clip_index.search(query_vector, top_k=args.top_k)

    # Build a lookup from (frame_index) -> actual image, so we can save the exact
    # frame that scored highly, not just re-decode from scratch.
    frame_by_index = {f.frame_index: f.image for f in sampled_frames}

    print(f"\nSaving top {len(hits)} flagged frames to {args.out_dir}/ ...")
    safe_query = "".join(c if c.isalnum() else "_" for c in args.query)[:40]

    for rank, (score, hit) in enumerate(zip(scores, hits)):
        image = frame_by_index.get(hit["frame_index"])
        if image is None:
            continue
        filename = f"{rank:02d}_score{score:.3f}_t{hit['timestamp_sec']:.1f}s_{safe_query}.jpg"
        out_path = os.path.join(args.out_dir, filename)
        cv2.imwrite(out_path, image)
        print(f"  [{rank}] score={score:.4f}  t={hit['timestamp_sec']:.2f}s  -> {out_path}")

    print("\nDone. Open the images above to visually check what got flagged.")


if __name__ == "__main__":
    main()