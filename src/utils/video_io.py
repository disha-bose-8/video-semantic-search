"""
video_io.py

moviepy-based utilities for extracting a video clip given start/end timestamps,
and other small video I/O helpers. moviepy is imported lazily so this module
stays cheap to import from places that only need e.g. get_video_duration()
via opencv.
"""

from __future__ import annotations

import os
from typing import Optional


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    codec: str = "libx264",
    audio_codec: str = "aac",
) -> str:
    """
    Extracts the [start_sec, end_sec] sub-clip of video_path and writes it to
    output_path. Returns output_path on success.
    """
    # moviepy 2.0+ removed the `.editor` submodule (VideoFileClip now lives at
    # the top-level `moviepy` package). moviepy 1.x still needs `.editor`.
    # Try the new import first, fall back to the old one.
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be > start_sec ({start_sec})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        start_sec = max(0.0, start_sec)
        end_sec = min(duration, end_sec)

        # moviepy 2.0+ renamed .subclip() to .subclipped(); support both.
        if hasattr(clip, "subclipped"):
            subclip = clip.subclipped(start_sec, end_sec)
        else:
            subclip = clip.subclip(start_sec, end_sec)

        subclip.write_videofile(
            output_path,
            codec=codec,
            audio_codec=audio_codec,
            logger=None,  # suppress moviepy's verbose progress bar in server contexts
        )

    return output_path


def get_video_duration(video_path: str) -> float:
    """Returns video duration in seconds using opencv (cheap, no moviepy needed)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames / fps
    finally:
        cap.release()


def make_clip_output_path(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str = "data/clips_extracted",
) -> str:
    """Deterministic output filename for an extracted clip, e.g. cam1_10.0-15.0.mp4"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"{base}_{start_sec:.1f}-{end_sec:.1f}.mp4"
    return os.path.join(output_dir, filename)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python video_io.py <video_path> <start_sec> <end_sec>")
        sys.exit(1)

    video_path = sys.argv[1]
    start_sec = float(sys.argv[2])
    end_sec = float(sys.argv[3])

    out_path = make_clip_output_path(video_path, start_sec, end_sec)
    result = extract_clip(video_path, start_sec, end_sec, out_path)
    print(f"Extracted clip written to: {result}")