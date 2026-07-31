import os
import requests

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
        try:
            # Envoi SANS header Authorization pour ne pas envoyer la cle Groq a HF
            response = requests.post(SENTIMENT_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                predictions = result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else result
                
                if isinstance(predictions, list) and len(predictions) > 0:
                    top_pred = max(predictions, key=lambda x: x.get("score", 0.0))
                    raw_label = str(top_pred.get("label", ""))
                    score = float(top_pred.get("score", 0.85))

                    return {
                        "sentiment": self._map_label(raw_label),
                        "confidence": round(score, 4)
                    }

        except Exception as e:
            print(f"[Warning Sentiment HF] : {e}")

        # Sécurité : retourne une valeur cohérente si l'API HF met du temps
        return {"sentiment": "neutre", "confidence": 0.85}