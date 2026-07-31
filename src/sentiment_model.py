from transformers import pipeline

class SentimentModel:
    def __init__(self):
        print("Chargement du modèle Sentiment CamemBERT...")
        # I. Modèle local Hugging Face
        self.classifier = pipeline(
            "text-classification",
            model="cmarkea/distilcamembert-base-sentiment",
            # Récupère tous les scores
            top_k=None  
        )

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

        try:
            results = self.classifier(text)[0]
            
            # II. Extraction dynamique du score max
            
            top_pred = max(results, key=lambda x: x["score"])
            
            return {
                "sentiment": self._map_label(top_pred["label"]),
                "confidence": round(float(top_pred["score"]), 4) # Dynamique [0.0 - 1.0]
            }
        except Exception as e:
            print(f"[Erreur Sentiment Local] : {e}")
            return {"sentiment": "neutre", "confidence": 0.0}