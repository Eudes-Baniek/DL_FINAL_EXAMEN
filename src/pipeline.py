import os
from src.preprocessing import preprocess_audio
from src.asr_model import ASRModel
from src.sentiment_model import SentimentModel

class AudioSentimentPipeline:
    def __init__(self):
        self.asr = ASRModel()
        self.sentiment = SentimentModel()

    def predict(self, audio_path: str) -> dict:
        processed_path = None
        try:
            # 1. Prétraitement (16kHz, mono)
            processed_path = preprocess_audio(audio_path)

            # 2. Transcription ASR
            transcription = self.asr.transcribe(processed_path)

            if not transcription:
                return {
                    "transcription": "",
                    "sentiment": "neutre",
                    "confidence": 0.0
                }

            # 3. Sentiment NLP (CamemBERT)
            sentiment_res = self.sentiment.predict(transcription)

            return {
                "transcription": transcription,
                "sentiment": sentiment_res["sentiment"],
                "confidence": sentiment_res["confidence"]
            }

        finally:
            # Nettoyage du fichier temporaire
            if processed_path and os.path.exists(processed_path):
                os.remove(processed_path)