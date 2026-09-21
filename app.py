import shutil
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from match_face import find_top_matches

app = FastAPI(title="Manga Face Match")


app.mount("/panels", StaticFiles(directory="manga_panels"), name="panels")

# Serve the frontend itself
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /static/index.html for the web UI, or POST to /match directly"}

@app.post("/match")
async def match(file: UploadFile = File(...), top_k: int = 1):
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        matches = find_top_matches(tmp_path, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp_path)

    results = [
        {"filename": fname, "emotion": emotion, "similarity": float(score)}
        for fname, emotion, score in matches
    ]

    return JSONResponse(content={"matches": results})