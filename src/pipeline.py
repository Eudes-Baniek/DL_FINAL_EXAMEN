from src.asr_model import ASRModel
from src.sentiment_model import SentimentModel

class AudioSentimentPipeline:
    def __init__(self):
        self.asr = ASRModel()
        self.sentiment = SentimentModel()

    def predict(self, audio_path: str) -> dict:
        # 1. ASR via API
        transcription = self.asr.transcribe(audio_path)

        # 2. Sentiment via API
        if transcription.strip():
            sentiment_res = self.sentiment.predict(transcription)
        else:
            sentiment_res = {"sentiment": "neutre", "confidence": 0.0}

        return {
            "transcription": transcription,
            "sentiment": sentiment_res["sentiment"],
            "confidence": sentiment_res["confidence"]
        }