"""
frame_sampler.py

Samples frames from a video at a fixed time interval (every N seconds).
No heavy ML deps here — only opencv — so this module is cheap to import
from anywhere in the pipeline (including the FastAPI backend at startup).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2


@dataclass
class SampledFrame:
    """A single sampled frame plus where it came from in the source video."""

    video_path: str
    frame_index: int          # absolute frame index in the source video
    timestamp_sec: float      # timestamp of this frame in seconds
    image: "cv2.Mat"          # BGR numpy array (as returned by cv2.VideoCapture.read())


class FrameSampler:
    """
    Samples frames from a video file every `interval_sec` seconds.

    Usage:
        sampler = FrameSampler(interval_sec=1.0)
        for frame in sampler.sample("data/raw_videos/cam1.mp4"):
            ...
    """

    def __init__(self, interval_sec: float = 1.0):
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        self.interval_sec = interval_sec

    def sample(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Iterator[SampledFrame]:
        """
        Yields SampledFrame objects at self.interval_sec spacing, in time order.

        start_sec/end_sec let you sample only a sub-range of the video
        (useful later for re-indexing a single clip range instead of a
        whole file).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0  # sane fallback for weird containers

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if total_frames > 0 else None

            if end_sec is None:
                end_sec = duration_sec if duration_sec is not None else float("inf")

            step_frames = max(1, round(self.interval_sec * fps))
            start_frame = max(0, round(start_sec * fps))
            end_frame = (
                total_frames - 1
                if duration_sec is not None
                else float("inf")
            )
            if end_sec != float("inf"):
                end_frame = min(end_frame, round(end_sec * fps))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            next_wanted_index = start_frame

            while True:
                if frame_index > end_frame:
                    break

                ok, image = cap.read()
                if not ok:
                    break

                if frame_index == next_wanted_index:
                    timestamp_sec = frame_index / fps
                    yield SampledFrame(
                        video_path=video_path,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image=image,
                    )
                    next_wanted_index += step_frames

                frame_index += 1
        finally:
            cap.release()

    def count_expected_samples(self, video_path: str) -> int:
        """Quick estimate of how many frames sample() will yield, without decoding."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_frames = max(1, round(self.interval_sec * fps))
            return max(0, (total_frames // step_frames) + 1)
        finally:
            cap.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_sampler.py <video_path> [interval_sec]")
        sys.exit(1)

    video_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    sampler = FrameSampler(interval_sec=interval)
    n = 0
    for f in sampler.sample(video_path):
        print(f"frame_index={f.frame_index} t={f.timestamp_sec:.2f}s shape={f.image.shape}")
        n += 1
    print(f"Total sampled frames: {n}")