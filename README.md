# Video Semantic Search

Natural-language search over video content - find moments in a video using 
plain English queries, without any manual tagging or keyword metadata.

## How it works

1. **Frame extraction** - video is sampled into frames at [ rate/strategy ]
2. **Embedding** - each frame is encoded using CLIP (ViT-B-32-quickgelu), 
   mapping it into a shared image-text vector space
3. **Indexing** - frame embeddings are stored in a FAISS [ index type ] 
   index for fast similarity search
4. **Query** - a text query is encoded by CLIP's text encoder into the 
   same vector space, then FAISS returns the nearest matching frames
5. **Serving** - results are exposed via a FastAPI backend

## Why CLIP?

CLIP is trained so that an image and a text description of similar content 
land close together in the same embedding space - so no labeled video data 
or manual tags are needed for search to work.

## Example

Query: `"[ example query ]"` → correctly retrieves `[ what it found ]`, 
despite [ why this was a nontrivial match — no exact keyword overlap etc. ]

## Tech stack

CLIP (ViT-B-32-quickgelu) · FAISS · FastAPI · Python

## Status

Backend and search pipeline complete and verified end-to-end. 
Frontend: [ in progress / not yet started ]

## Setup

\`\`\`bash
pip install -r requirements.txt
[ run instructions ]
\`\`\`
