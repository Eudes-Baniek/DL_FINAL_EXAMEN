import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/cmarkea/distilcamembert-base-sentiment"

class SentimentModel:
    def _map_label(self, raw_label: str) -> str:
        label = raw_label.lower().strip()
        if "1 star" in label or "2 star" in label or "label_0" in label or "neg" in label:
            return "négatif"
        elif "3 star" in label or "label_1" in label or "neu" in label:
            return "neutre"
        elif "4 star" in label or "5 star" in label or "label_2" in label or "pos" in label:
            return "positif"
        return "neutre"

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        payload = {"inputs": text}
        response = requests.post(SENTIMENT_URL, headers=HEADERS, json=payload, timeout=20)

        if response.status_code != 200:
            raise RuntimeError(f"Code {response.status_code} - {response.text}")

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            predictions = result[0] if isinstance(result[0], list) else result
            top_pred = max(predictions, key=lambda x: x.get("score", 0.0))
            
            raw_label = top_pred.get("label", "")
            score = float(top_pred.get("score", 0.0))
            
            return {
                "sentiment": self._map_label(raw_label),
                "confidence": score
            }

        return {"sentiment": "neutre", "confidence": 0.0}