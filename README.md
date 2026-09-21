# Manga Face Match

Upload a photo of your face and get matched with the manga expression panel that most closely resembles it.

This isn't identity matching ("is this the same person"). It's **expression matching across domains**: mapping a real photo and a stylized manga panel into the same embedding space to find the closest emotional/expressive match, despite the two having completely different visual styles (photographic vs. line-art).

## Screenshots
<img width="764" height="702" alt="image" src="https://github.com/user-attachments/assets/3632c204-b975-439d-9f47-a23fca4ca384" />


## Demo

Upload a photo → the app detects your face → embeds it with CLIP → compares it against a dataset of manga expression panels → returns the top matches with similarity scores.

*(Add a screenshot or GIF of the web UI here once you have one — this is the single highest-impact addition for a GitHub README)*

## How it works

1. **Face detection & cropping** — MediaPipe locates the face in the uploaded photo and crops tightly around it, removing background noise that would otherwise pollute the embedding.
2. **Embedding** — The cropped face is passed through a CLIP vision encoder (`ViT-L-14`, LAION-pretrained checkpoint) and converted into a 768-dimensional vector representing its visual/expressive content.
3. **Matching** — That vector is compared via cosine similarity against a precomputed index of embeddings for ~450 manga expression panels spanning 7 emotion categories (shock, sad, happy, pleased, embarrassed, crying, angry).
4. **Ranking** — The closest matches are returned, along with their emotion label and similarity score.

The key idea: CLIP was trained on a huge, diverse mix of image-text pairs, so it learns to place images with similar *meaning* close together in vector space — even across very different visual styles. A real smiling face and a manga panel of a smiling character can end up with similar embeddings, despite sharing almost no pixel-level similarity.

## Tech stack

- **PyTorch** + **open_clip** — CLIP model for generating image embeddings
- **OpenCV** / **MediaPipe** — face detection and cropping
- **NumPy** — vector storage and cosine similarity computation
- **FastAPI** + **Uvicorn** — REST API serving the matching pipeline
- **Vanilla JS/HTML/CSS** — lightweight frontend for photo upload and result display
- **Docker** — containerized environment for reproducible setup

## Project structure

```
manga-face-match/
├── manga_panels/          # manga expression dataset, organized by emotion
│   ├── shock/
│   ├── sad/
│   ├── happy/
│   ├── pleased/
│   ├── embarrassed/
│   ├── crying/
│   └── angry/
├── static/
│   └── index.html         # web frontend
├── build_index.py         # generates embeddings for the manga dataset
├── match_face.py          # face detection + embedding + matching logic
├── app.py                 # FastAPI server
├── requirements.txt
├── Dockerfile
└── manga_embeddings.npz   # generated embedding index (not tracked in git)
```

## Running it locally

**Prerequisites:** Docker Desktop installed and running.

**1. Build the image**
```bash
docker build -t manga-face-match .
```

**2. Populate the dataset**
Add manga expression panels to `manga_panels/<emotion>/`, organized by emotion subfolder.

**3. Build the embedding index**
```bash
docker run --rm -v ${PWD}:/app manga-face-match python build_index.py
```

**4. Start the server**
```bash
docker run --rm -p 8000:8000 -v ${PWD}:/app manga-face-match
```

**5. Open the app**
```
http://localhost:8000/static/index.html
```

## API

**`POST /match`**

Upload an image file to get the top matching manga panels.

```bash
curl -X POST "http://localhost:8000/match?top_k=2" \
  -F "file=@your_photo.jpg"
```

Response:
```json
{
  "matches": [
    {"filename": "crying/028_298_713_232_232.png", "emotion": "crying", "similarity": 0.7886},
    {"filename": "angry/040_464_100_177_177.png", "emotion": "angry", "similarity": 0.6947}
  ]
}
```

Interactive API docs available at `http://localhost:8000/docs`.

## Design notes & tradeoffs

- **No manual emotion labeling required for matching** — the system works purely on visual embedding similarity, not a trained classifier, which makes it easy to extend to new expression categories without retraining.
- **Domain gap is the core technical challenge.** Standard CLIP checkpoints are trained mostly on photographic/web imagery, so a LAION-pretrained checkpoint (which includes more illustrated/anime-style content) was used instead of the default OpenAI weights to improve cross-domain matching.
- **Face cropping quality directly affects match quality**, since CLIP embeds whatever is in the frame — MediaPipe was chosen over a Haar cascade for tighter, more consistent crops.

## Potential improvements

- Text-based emotion anchoring using CLIP's text encoder as a secondary signal alongside image-to-image similarity
- Fine-tuning the embedding space with contrastive learning on manually paired photo/manga examples
- Expanding the manga dataset and balancing image counts per emotion category
- Deploying with a persistent GPU-backed inference endpoint for faster response times at scale


