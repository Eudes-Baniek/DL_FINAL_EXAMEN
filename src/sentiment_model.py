import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentModel:

    def __init__(
        self,
        model_name: str = "tblard/tf-allocine-camembert",
    ):  # ou un modèle 3 classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Utilisation d'un pipeline Hugging Face d'analyse de sentiment en français
        from transformers import pipeline

        self.pipe = pipeline(
            "text-classification",
            model="cmarkea/distilcamembert-base-sentiment",
            device=0 if self.device == "cuda" else -1,
        )

        # Mapping des labels si nécessaire
        self.label_mapping = {
            "LABEL_0": "négatif",
            "LABEL_1": "neutre",
            "LABEL_2": "positif",
            "POSITIVE": "positif",
            "NEGATIVE": "négatif",
            "NEUTRAL": "neutre",
            "1 star": "négatif",
            "2 stars": "négatif",
            "3 stars": "neutre",
            "4 stars": "positif",
            "5 stars": "positif",
        }

    def predict(self, text: str):
        #Analyse le sentiment d'un texte et renvoie le sentiment + score.
        
        if not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        result = self.pipe(text)[0]
        raw_label = result["label"]
        score = round(float(result["score"]), 4)

        sentiment = self.label_mapping.get(raw_label, raw_label.lower())

        return {"sentiment": sentiment, "confidence": score}