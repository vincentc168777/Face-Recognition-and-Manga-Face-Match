import torch
import open_clip
import numpy as np
import mediapipe as mp
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)
model = model.to(device).eval()

mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

def crop_face(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    img_array = np.array(pil_img)

    results = mp_face_detection.process(img_array)

    if not results.detections:
        raise ValueError("No face detected")

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = img_array.shape

    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    box_w = min(int(bbox.width * w), w - x)
    box_h = min(int(bbox.height * h), h - y)

    cropped = img_array[y:y + box_h, x:x + box_w]

    if cropped.size == 0:
        raise ValueError("Face crop resulted in an empty image")

    return Image.fromarray(cropped)

def embed_pil_image(pil_image):
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

def find_top_matches(photo_path, top_k=1):
    data = np.load("manga_embeddings.npz", allow_pickle=True)
    filenames = data["filenames"]
    emotions = data["emotions"]
    vectors = data["vectors"]  # shape: (num_panels, embedding_dim)

    face_img = crop_face(photo_path)
    query_vec = embed_pil_image(face_img)  # shape: (embedding_dim,)

    # sklearn expects 2D arrays: reshape query to (1, embedding_dim)
    similarities = cosine_similarity(query_vec.reshape(1, -1), vectors)[0]

    ranked = sorted(zip(filenames, emotions, similarities), key=lambda x: x[2], reverse=True)

    return ranked[:top_k]

if __name__ == "__main__":
    photo_path = "test_photo.jpg"
    matches = find_top_matches(photo_path, top_k=1)

    print("Top matches:")
    for fname, emotion, score in matches:
        print(f"  {fname} (emotion: {emotion}) — similarity: {score:.4f}")