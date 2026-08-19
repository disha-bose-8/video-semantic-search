"""
backend/app.py

FastAPI backend for ClipSeek v1.

Endpoints:
    GET  /health                    -> liveness check
    POST /index                     -> index a video file already on disk (or uploaded)
    POST /search                    -> natural-language text query -> ranked ClipRanges
    POST /extract_clip              -> extract an mp4 for a given video_path/start/end
    GET  /clips/{filename}          -> serve an extracted clip file

Run with:
    uvicorn backend.app:app --reload --port 8000

Design notes:
- The ClipEmbedder (which loads the torch/open_clip model) is created ONCE at
  startup and reused across requests, both for indexing and for embedding
  search queries — reloading it per-request would be very slow.
- The FAISS index is kept in memory (self.searcher) and reloaded from disk
  whenever /index adds new videos, so /search always reflects the latest data.
- Uploaded videos are saved under data/raw_videos/; extracted clips are
  written to data/clips_extracted/ and served back via /clips/{filename}.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make `src` importable when running via `uvicorn backend.app:app` from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.indexing.build_index import ClipIndex, build_index_from_video  # noqa: E402
from src.query.search import VideoSearcher, ClipRange  # noqa: E402
from src.utils.video_io import extract_clip, make_clip_output_path, get_video_duration  # noqa: E402

RAW_VIDEOS_DIR = os.path.join(REPO_ROOT, "data", "raw_videos")
CLIPS_DIR = os.path.join(REPO_ROOT, "data", "clips_extracted")
INDEX_DIR = os.path.join(REPO_ROOT, "data", "index")

os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------- request/response models ----------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    merge_gap_sec: float = 2.0
    pad_sec: float = 0.5
    min_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[dict]


class IndexRequest(BaseModel):
    video_path: str  # path relative to data/raw_videos/, or absolute
    interval_sec: float = 1.0


class IndexResponse(BaseModel):
    video_path: str
    num_frames_indexed: int
    total_index_size: int


class ExtractClipRequest(BaseModel):
    video_path: str
    start_sec: float
    end_sec: float


class ExtractClipResponse(BaseModel):
    clip_filename: str
    clip_url: str


# ---------- app + shared state ----------

app = FastAPI(title="ClipSeek API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    """Holds the (lazily-loaded, heavy) embedder + searcher shared across requests."""

    def __init__(self):
        self.embedder = None  # ClipEmbedder, created on first use
        self.searcher: Optional[VideoSearcher] = None

    def get_embedder(self):
        if self.embedder is None:
            from src.indexing.embed_frames import ClipEmbedder

            print("Loading CLIP model (first request)...")
            self.embedder = ClipEmbedder()
            print(f"CLIP model loaded on {self.embedder.device}")
        return self.embedder

    def reload_searcher(self):
        """Reload the FAISS index from disk, reusing the already-loaded embedder."""
        if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            self.searcher = None
            return
        self.searcher = VideoSearcher(index_dir=INDEX_DIR, embedder=self.get_embedder())


state = AppState()


@app.on_event("startup")
def startup_event():
    # Try to load an existing index if one is already on disk from a previous run.
    try:
        state.reload_searcher()
        if state.searcher is not None:
            print(f"Loaded existing index with {len(state.searcher.index)} vectors.")
    except Exception as e:
        print(f"No existing index loaded at startup: {e}")


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": state.searcher is not None,
        "index_size": len(state.searcher.index) if state.searcher else 0,
    }


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a raw video into data/raw_videos/. Returns the path to pass to /index."""
    dest_path = os.path.join(RAW_VIDEOS_DIR, file.filename)
    with open(dest_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return {"video_path": file.filename, "saved_to": dest_path}


@app.post("/index", response_model=IndexResponse)
def index_video(req: IndexRequest):
    """
    Index a video that already exists on disk (either an absolute path, or a
    filename relative to data/raw_videos/, e.g. after /upload_video).
    """
    video_path = req.video_path
    if not os.path.isabs(video_path):
        video_path = os.path.join(RAW_VIDEOS_DIR, video_path)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    embedder = state.get_embedder()

    # If an index already exists, append to it; otherwise create fresh.
    from src.indexing.frame_sampler import FrameSampler

    sampler = FrameSampler(interval_sec=req.interval_sec)
    sampled_frames = list(sampler.sample(video_path))
    if not sampled_frames:
        raise HTTPException(status_code=400, detail="No frames could be sampled from this video")

    frame_embeddings = embedder.embed_sampled_frames(sampled_frames)

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        clip_index = ClipIndex.load(INDEX_DIR)
    else:
        clip_index = ClipIndex(embed_dim=embedder.embed_dim)

    clip_index.add_frame_embeddings(frame_embeddings)
    clip_index.save(INDEX_DIR)

    state.reload_searcher()

    return IndexResponse(
        video_path=video_path,
        num_frames_indexed=len(frame_embeddings),
        total_index_size=len(clip_index),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if state.searcher is None:
        raise HTTPException(
            status_code=400,
            detail="No index available yet. Index at least one video via /index first.",
        )

    ranges: List[ClipRange] = state.searcher.search(
        query_text=req.query,
        top_k=req.top_k,
        merge_gap_sec=req.merge_gap_sec,
        pad_sec=req.pad_sec,
        min_score=req.min_score,
    )

    return SearchResponse(query=req.query, results=[r.to_dict() for r in ranges])


@app.post("/extract_clip", response_model=ExtractClipResponse)
def extract_clip_endpoint(req: ExtractClipRequest):
    video_path = req.video_path
    if not os.path.isabs(video_path):
        video_path = os.path.join(RAW_VIDEOS_DIR, video_path)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    output_path = make_clip_output_path(
        video_path, req.start_sec, req.end_sec, output_dir=CLIPS_DIR
    )

    try:
        extract_clip(video_path, req.start_sec, req.end_sec, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip extraction failed: {e}")

    filename = os.path.basename(output_path)
    return ExtractClipResponse(clip_filename=filename, clip_url=f"/clips/{filename}")


@app.get("/clips/{filename}")
def get_clip(filename: str):
    path = os.path.join(CLIPS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)