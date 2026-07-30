from src.asr_model import ASRModel
from src.audio_utils import validate_and_load_audio
from src.sentiment_model import SentimentModel


class AudioSentimentPipeline:

    def __init__(self):
        print("Chargement du modèle ASR...")
        self.asr = ASRModel()
        print("Chargement du modèle de Sentiment...")
        self.sentiment = SentimentModel()

    def predict(self, audio_path: str) -> dict:
        
        # Pipeline complet : Audio -> Prétraitement -> ASR -> Sentiment -> JSON
        # 1. Validation & Chargement audio (16kHz, mono)
        
        audio_data, sr = validate_and_load_audio(audio_path)

        # 2. ASR (Speech to Text)
        transcription = self.asr.transcribe(audio_data, sampling_rate=sr)

        # 3. Sentiment Analysis
        sentiment_res = self.sentiment.predict(transcription)

        # 4. Résultat unifié
        return {
            "transcription": transcription,
            "sentiment": sentiment_res["sentiment"],
            "confidence": sentiment_res["confidence"],
        }