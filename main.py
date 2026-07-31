from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from src.pipeline import AudioSentimentPipeline

app = FastAPI(
    title="API de Détection de Sentiment Vocal",
    description="API REST pour la transcription et l'analyse de sentiment d'appels vocaux",
    version="1.0"
)

pipeline = AudioSentimentPipeline()

@app.get("/")
def root():
    return {"status": "API en ligne", "endpoint": "/predict"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not (file.filename.endswith(".wav") or file.filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Format audio non supporté. Utiliser .wav ou .mp3")

    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.predict(temp_filename)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)