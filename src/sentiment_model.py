import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/cmarkea/distilcamembert-base-sentiment"

class SentimentModel:
    def predict(self, text: str) -> dict:
        # Analyse le sentiment d'un texte via l'API Inference de Hugging Face.
        if not text or not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        payload = {"inputs": text}
        try:
            response = requests.post(SENTIMENT_URL, headers=HEADERS, json=payload, timeout=20)
            if response.status_code != 200:
                raise RuntimeError(f"Erreur Sentiment (Code {response.status_code}) : {response.text}")

            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                top_pred = max(predictions, key=lambda x: x.get("score", 0.0))
                return {
                    "sentiment": top_pred.get("label", "neutre"),
                    "confidence": float(top_pred.get("score", 0.0))
                }
        except Exception as e:
            print(f"Exception lors de l'analyse du sentiment : {e}")
            
        return {"sentiment": "neutre", "confidence": 0.0}