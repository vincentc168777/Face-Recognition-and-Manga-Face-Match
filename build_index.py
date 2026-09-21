import os
import torch
import open_clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)
model = model.to(device).eval()

MANGA_DIR = "manga_panels"  #subfolders for faces
OUTPUT_FILE = "manga_embeddings.npz"

def embed_image(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

def build_index():
    filenames = []
    emotions = []
    vectors = []

    for emotion_folder in sorted(os.listdir(MANGA_DIR)):
        folder_path = os.path.join(MANGA_DIR, emotion_folder)

        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder_path, fname)
                vec = embed_image(path)

                filenames.append(f"{emotion_folder}/{fname}")
                emotions.append(emotion_folder)
                vectors.append(vec)
                print(f"Embedded: {emotion_folder}/{fname}")

    vectors = np.array(vectors)
    np.savez(OUTPUT_FILE, filenames=filenames, emotions=emotions, vectors=vectors)
    print(f"\nSaved {len(filenames)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_index()