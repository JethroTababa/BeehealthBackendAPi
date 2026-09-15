import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline


MODEL_REPO = "troyskie/Beewatch-hive-classifier"

app = FastAPI(
    title="BeeWatch Pro Backend",
    description="BeeWatch Pro AI Hive Acoustic Classification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading BeeWatch AST model...")

pipe = pipeline(
    "audio-classification",
    model=MODEL_REPO
)

print("BeeWatch AST Model Loaded")


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "BeeWatch Backend Running",
        "model": MODEL_REPO
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_REPO
    }


@app.post("/classify")
async def classify(audio_file: UploadFile = File(...)):

    temp_path = f"/tmp/{audio_file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        print("🐝 BeeWatch Audio Received:", audio_file.filename)

        results = pipe(temp_path)

        prediction = max(
            results,
            key=lambda x: x["score"]
        )

        label = prediction["label"]
        confidence = prediction["score"]

        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
            "all_predictions": [
                {
                    "label": result["label"],
                    "score": round(float(result["score"]), 4)
                }
                for result in results
            ]
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)