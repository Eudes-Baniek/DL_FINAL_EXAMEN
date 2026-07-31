import os
import requests

SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/cmarkea/distilcamembert-base-sentiment"

class SentimentModel:
    def _map_label(self, raw_label: str) -> str:
        """Mappe les labels vers : positif, neutre, négatif."""
        label = raw_label.lower().strip()
        if "1 star" in label or "2 star" in label or "label_0" in label or "neg" in label:
            return "négatif"
        elif "3 star" in label or "label_1" in label or "neu" in label:
            return "neutre"
        elif "4 star" in label or "5 star" in label or "label_2" in label or "pos" in label:
            return "positif"
        return "neutre"

    def predict(self, text: str) -> dict:
        """Analyse le sentiment via l'API publique Hugging Face et extrait le score exact."""
        if not text or not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        payload = {"inputs": text}
        try:
            response = requests.post(SENTIMENT_URL, json=payload, timeout=20)

            if response.status_code != 200:
                print(f"[Warning Sentiment] Code {response.status_code} - {response.text}")
                return {"sentiment": "neutre", "confidence": 0.0}

            result = response.json()

            # Extraction robuste des prédictions
            predictions = []
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list) and len(result[0]) > 0:
                    predictions = result[0]
                else:
                    predictions = result

            if predictions:
                # Trouver la prédiction avec le meilleur score
                top_pred = max(predictions, key=lambda x: x.get("score", 0.0))
                
                raw_label = str(top_pred.get("label", ""))
                score = float(top_pred.get("score", 0.0))

                return {
                    "sentiment": self._map_label(raw_label),
                    "confidence": round(score, 4)  # Arrondi propre à 4 décimales
                }

        except Exception as e:
            print(f"[Erreur Sentiment Exec] : {e}")

        return {"sentiment": "neutre", "confidence": 0.0}