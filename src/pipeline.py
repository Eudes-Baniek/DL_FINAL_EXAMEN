from src.asr_model import ASRModel
from src.sentiment_model import SentimentModel

class AudioSentimentPipeline:
    def __init__(self):
        self.asr = ASRModel()
        self.sentiment = SentimentModel()

    def predict(self, audio_path: str) -> dict:
        # 1. Transcription ASR
        transcription = self.asr.transcribe(audio_path)

        if not transcription:
            return {
                "transcription": "",
                "sentiment": "neutre",
                "confidence": 0.0
            }

        # 2. Analyse de sentiment
        sentiment_res = self.sentiment.predict(transcription)

        return {
            "transcription": transcription,
            "sentiment": sentiment_res["sentiment"],
            "confidence": sentiment_res["confidence"]
        }