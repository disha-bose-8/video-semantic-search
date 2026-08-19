"""
scripts/test_real_video.py

Full end-to-end ClipSeek test against a REAL video file, using the REAL
downloaded CLIP weights. Run this from the repo root with your venv active:

    python scripts/test_real_video.py --video data/raw_videos/YOUR_VIDEO.mp4 --query "a person walking"

What it does:
    1. Samples frames from the video every --interval seconds (default 1s)
    2. Embeds every sampled frame with CLIP (ViT-B-32-quickgelu, openai weights)
    3. Builds a FAISS index in memory (and saves it to --index_dir)
    4. Embeds your text --query and searches the index
    5. Prints ranked ClipRange results (start/end timestamps + score)
    6. Optionally extracts the top-N matching clips as actual .mp4 files

This is exactly what the /index and /search FastAPI endpoints do internally —
running it standalone first makes it much easier to sanity-check quality and
tune --interval / --merge_gap / --pad before wiring up the backend+frontend.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.indexing.frame_sampler import FrameSampler
from src.indexing.embed_frames import ClipEmbedder
from src.indexing.build_index import ClipIndex
from src.query.search import group_into_clip_ranges
from src.utils.video_io import extract_clip, make_clip_output_path, get_video_duration


def main():
    parser = argparse.ArgumentParser(description="ClipSeek real-video end-to-end test")
    parser.add_argument("--video", required=True, help="Path to a real video file")
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames")
    parser.add_argument("--top_k", type=int, default=10, help="How many frame hits to retrieve")
    parser.add_argument("--merge_gap", type=float, default=2.0, help="Seconds gap to merge hits into one clip range")
    parser.add_argument("--pad", type=float, default=0.5, help="Seconds to pad each clip range")
    parser.add_argument("--index_dir", default="data/index", help="Where to save the FAISS index")
    parser.add_argument("--extract_top", type=int, default=0, help="Extract this many top clips as .mp4 files")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)

    duration = get_video_duration(args.video)
    print(f"Video: {args.video} ({duration:.1f}s)")

    # ---- 1. Sample frames ----
    print(f"\n[1/4] Sampling frames every {args.interval}s ...")
    t0 = time.time()
    sampler = FrameSampler(interval_sec=args.interval)
    sampled_frames = list(sampler.sample(args.video))
    print(f"  Sampled {len(sampled_frames)} frames in {time.time()-t0:.1f}s")

    if not sampled_frames:
        print("ERROR: no frames sampled — check the video file / interval.")
        sys.exit(1)

    # ---- 2. Embed frames ----
    print(f"\n[2/4] Loading CLIP (ViT-B-32-quickgelu, openai) and embedding frames ...")
    t0 = time.time()
    embedder = ClipEmbedder()
    frame_embeddings = embedder.embed_sampled_frames(sampled_frames)
    print(f"  Embedded {len(frame_embeddings)} frames in {time.time()-t0:.1f}s "
          f"(device={embedder.device}, dim={embedder.embed_dim})")

    # ---- 3. Build + save index ----
    print(f"\n[3/4] Building FAISS index ...")
    clip_index = ClipIndex(embed_dim=embedder.embed_dim)
    clip_index.add_frame_embeddings(frame_embeddings)
    clip_index.save(args.index_dir)
    print(f"  Index built with {len(clip_index)} vectors, saved to {args.index_dir}/")

    # ---- 4. Search ----
    print(f"\n[4/4] Searching for: \"{args.query}\"")
    query_vector = embedder.embed_text(args.query)
    scores, hits = clip_index.search(query_vector, top_k=args.top_k)

    print(f"\nRaw frame hits (top {len(hits)}):")
    for score, hit in zip(scores, hits):
        print(f"  score={score:.4f}  t={hit['timestamp_sec']:.2f}s  frame_idx={hit['frame_index']}")

    ranges = group_into_clip_ranges(scores, hits, merge_gap_sec=args.merge_gap, pad_sec=args.pad)

    print(f"\nGrouped clip ranges ({len(ranges)}):")
    for i, r in enumerate(ranges):
        d = r.to_dict()
        print(f"  [{i}] score={d['score']:.4f}  {d['start_sec']:.1f}s -> {d['end_sec']:.1f}s  "
              f"({d['num_frame_hits']} frame hits)")

    # ---- optional: extract top-N clips as real mp4 files ----
    if args.extract_top > 0 and ranges:
        print(f"\nExtracting top {min(args.extract_top, len(ranges))} clip(s) ...")
        os.makedirs("data/clips_extracted", exist_ok=True)
        for r in ranges[: args.extract_top]:
            out_path = make_clip_output_path(args.video, r.start_sec, r.end_sec)
            try:
                extract_clip(args.video, r.start_sec, r.end_sec, out_path)
                print(f"  Wrote {out_path}")
            except Exception as e:
                print(f"  FAILED to extract {r.start_sec:.1f}-{r.end_sec:.1f}s: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()