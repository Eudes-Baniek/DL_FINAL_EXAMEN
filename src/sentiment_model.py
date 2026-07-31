import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

class SentimentModel:
    def __init__(self):
        self.client = InferenceClient(
            model="cmarkea/distilcamembert-base-sentiment",
            token=HF_TOKEN
        )

    def _map_label(self, raw_label: str) -> str:
        """Mappe les labels bruts du modèle vers : positif, neutre, negatif."""
        label = raw_label.lower().strip()
        
        # Cas 1 : Labels type étoiles (1 star -> négatif, 3 stars -> neutre, 5 stars -> positif)
        if "1 star" in label or "2 star" in label or "LABEL_0" in label:
            return "négatif"
        elif "3 star" in label or "LABEL_1" in label:
            return "neutre"
        elif "4 star" in label or "5 star" in label or "LABEL_2" in label:
            return "positif"
        
        # Cas 2 : Labels texte explicite (ex: POSITIF, NEGATIF)
        if "pos" in label:
            return "positif"
        elif "neg" in label:
            return "négatif"
        
        return "neutre"

    def predict(self, text: str) -> dict:
        """Analyse le sentiment via InferenceClient et retourne positif, neutre ou négatif."""
        if not text or not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        try:
            response = self.client.text_classification(text)
            
            if response and len(response) > 0:
                # Récupère la prédiction avec le score le plus élevé
                top_pred = max(
                    response, 
                    key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else getattr(x, "score", 0.0)
                )
                
                raw_label = top_pred.get("label") if isinstance(top_pred, dict) else getattr(top_pred, "label", "")
                score = top_pred.get("score") if isinstance(top_pred, dict) else getattr(top_pred, "score", 0.0)
                
                mapped_sentiment = self._map_label(raw_label)
                
                return {
                    "sentiment": mapped_sentiment,
                    "confidence": float(score)
                }

        except Exception as e:
            # Utile pour déboguer sur les logs Render
            print(f"[ERREUR SentimentModel API] : {e}")
            raise RuntimeError(f"Erreur d'analyse du sentiment : {str(e)}")

        return {"sentiment": "neutre", "confidence": 0.0}