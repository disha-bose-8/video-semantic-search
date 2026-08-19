"""
search.py

Given a natural-language text query:
  1. Embed the query with CLIP (same embedding space as indexed frames).
  2. Search the FAISS index for the most similar frames.
  3. Group nearby frame hits (same video, close in time) into contiguous
     "clip ranges" with a start/end timestamp and an aggregate score.

This is the module the FastAPI /search endpoint calls into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.indexing.build_index import ClipIndex


@dataclass
class ClipRange:
    """A contiguous range of time in a video that matched the query."""

    video_path: str
    start_sec: float
    end_sec: float
    score: float                 # aggregate (max) similarity score across frames in the range
    num_frame_hits: int          # how many individual frame hits were merged into this range
    frame_timestamps: List[float]  # raw timestamps of the merged frame hits, for debugging/UI

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "score": round(float(self.score), 4),
            "num_frame_hits": self.num_frame_hits,
            "frame_timestamps": [round(t, 2) for t in self.frame_timestamps],
        }


def group_into_clip_ranges(
    scores: List[float],
    hits: List[dict],
    merge_gap_sec: float = 2.0,
    pad_sec: float = 0.5,
) -> List[ClipRange]:
    """
    Merge individual (score, metadata) frame hits into contiguous ClipRange objects.

    Hits from the same video whose timestamps are within `merge_gap_sec` of each
    other get merged into a single range. `pad_sec` extends each range's
    start/end slightly so the resulting clip isn't a single-frame sliver.

    Ranges are returned sorted by descending score (best match first).
    """
    if not hits:
        return []

    # Group hits by video, then sort each group by timestamp.
    by_video: dict = {}
    for score, hit in zip(scores, hits):
        by_video.setdefault(hit["video_path"], []).append((hit["timestamp_sec"], score))

    ranges: List[ClipRange] = []

    for video_path, points in by_video.items():
        points.sort(key=lambda p: p[0])  # sort by timestamp

        current_group = [points[0]]
        for ts, score in points[1:]:
            last_ts = current_group[-1][0]
            if ts - last_ts <= merge_gap_sec:
                current_group.append((ts, score))
            else:
                ranges.append(_group_to_range(video_path, current_group, pad_sec))
                current_group = [(ts, score)]

        ranges.append(_group_to_range(video_path, current_group, pad_sec))

    ranges.sort(key=lambda r: r.score, reverse=True)
    return ranges


def _group_to_range(video_path: str, group: List[tuple], pad_sec: float) -> ClipRange:
    timestamps = [ts for ts, _ in group]
    scores = [score for _, score in group]

    start_sec = max(0.0, min(timestamps) - pad_sec)
    end_sec = max(timestamps) + pad_sec

    return ClipRange(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        score=max(scores),
        num_frame_hits=len(group),
        frame_timestamps=timestamps,
    )


class VideoSearcher:
    """
    Ties together a loaded ClipIndex + a ClipEmbedder for text-query search.

    Usage:
        searcher = VideoSearcher(index_dir="data/index")
        ranges = searcher.search("a person wearing a red jacket", top_k=20)
    """

    def __init__(self, index_dir: str, embedder: Optional[object] = None):
        self.index_dir = index_dir
        self.index = ClipIndex.load(index_dir)

        if embedder is None:
            # Lazy import so this module doesn't require torch unless search() is called.
            from src.indexing.embed_frames import ClipEmbedder

            embedder = ClipEmbedder()
        self.embedder = embedder

    def search(
        self,
        query_text: str,
        top_k: int = 20,
        merge_gap_sec: float = 2.0,
        pad_sec: float = 0.5,
        min_score: Optional[float] = None,
    ) -> List[ClipRange]:
        query_vector = self.embedder.embed_text(query_text)
        scores, hits = self.index.search(query_vector, top_k=top_k)

        if min_score is not None:
            filtered = [(s, h) for s, h in zip(scores, hits) if s >= min_score]
            scores = [s for s, _ in filtered]
            hits = [h for _, h in filtered]

        return group_into_clip_ranges(scores, hits, merge_gap_sec=merge_gap_sec, pad_sec=pad_sec)


if __name__ == "__main__":
    # Mock smoke test for group_into_clip_ranges — no FAISS/torch needed.
    print("Running mock smoke test for group_into_clip_ranges...")

    mock_scores = [0.9, 0.85, 0.4, 0.88, 0.3]
    mock_hits = [
        {"video_path": "cam1.mp4", "frame_index": 0, "timestamp_sec": 10.0},
        {"video_path": "cam1.mp4", "frame_index": 1, "timestamp_sec": 11.0},
        {"video_path": "cam1.mp4", "frame_index": 2, "timestamp_sec": 40.0},
        {"video_path": "cam2.mp4", "frame_index": 0, "timestamp_sec": 5.0},
        {"video_path": "cam2.mp4", "frame_index": 1, "timestamp_sec": 25.0},
    ]

    ranges = group_into_clip_ranges(mock_scores, mock_hits, merge_gap_sec=2.0, pad_sec=0.5)
    for r in ranges:
        print(r.to_dict())

    assert len(ranges) == 4, f"expected 4 ranges, got {len(ranges)}"
    print("Smoke test passed.")