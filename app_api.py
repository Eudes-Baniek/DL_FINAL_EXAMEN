import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(
    title="API de Détection de Sentiment Audio",
    description="API REST pour la transcription audio et l'analyse de sentiment (ASR -> NLP)"
)

# Configuration de l'Inference Client Hugging Face

render-deploy = os.getenv("render-deploy")
client = InferenceClient(token=render-deploy)

ASR_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
SENTIMENT_MODEL = "cmarkea/distilcamembert-base-sentiment"

class PredictionResponse(BaseModel):
    transcription: str
    sentiment: str
    score: float

@app.get("/")
def read_root():
    return {"message": "API de détection de sentiment audio opérationnelle."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    # Vérification du format audio
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Format audio non supporté. Utilisez .wav ou .mp3")

    try:
        # Lecture des octets du fichier audio envoyé
        
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Le fichier audio fourni est vide.")

        # 1. ASR - Transcription via Wav2Vec 2.0
        
        asr_response = client.automatic_speech_recognition(
            audio_bytes,
            model=ASR_MODEL
        )
        transcription = asr_response.get("text", "").strip()

        if not transcription:
            return PredictionResponse(
                transcription="[Audio inaudible ou silencieux]",
                sentiment="neutre",
                score=0.0
            )

        # 2. NLP - Analyse de sentiment via CamemBERT
        
        sentiment_response = client.text_classification(
            transcription,
            model=SENTIMENT_MODEL
        )

        # Extraction du meilleur sentiment et son score
        top_sentiment = max(sentiment_response, key=lambda x: x.get("score", 0.0))
        
        return PredictionResponse(
            transcription=transcription,
            sentiment=top_sentiment.get("label", "inconnu"),
            score=round(float(top_sentiment.get("score", 0.0)), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")