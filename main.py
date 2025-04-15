from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

# CORS per permettere richieste dal frontend GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puoi restringere questo in produzione
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica i modelli una volta sola
detection_model = YOLO("D_best.pt")
classification_model = YOLO("C_best.pt")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detection
    results = detection_model.predict(image)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return {"error": "Nessuna lesione rilevata"}

    # Prendi la prima bounding box
    box = boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    # Ritaglia l'immagine
    cropped = image.crop((x1, y1, x2, y2))

    # Classificazione
    classification_result = classification_model.predict(cropped, verbose=False)[0]
    probs = classification_result.probs

    if probs is None:
        return {"error": "Classificazione fallita"}

    class_probs = {
        classification_model.names[i]: float(prob)
        for i, prob in enumerate(probs.data.cpu().numpy())
    }

    return {
        "bbox": [x1, y1, x2, y2],
        "class_probabilities": class_probs
    }
