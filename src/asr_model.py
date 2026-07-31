import os
import time
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

class ASRModel:
    def __init__(self):
        # Utilisation de l'API d'Inference HF
        self.client = InferenceClient(
            model="openai/whisper-small",
            token=HF_TOKEN
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcrit l'audio via l'InferenceClient d'Hugging Face."""
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Fichier audio introuvable ou chemin invalide.")

        # Lecture du fichier audio en binaire pour transmission directe
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le fichier audio : {str(e)}")

        max_retries = 3
        last_error = ""

        for attempt in range(max_retries):
            try:
                # Passage des octets bruts directement à la méthode
                result = self.client.automatic_speech_recognition(audio_bytes)
                
                if isinstance(result, dict):
                    return result.get("text", "").strip()
                elif hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()

            except Exception as e:
                last_error = str(e) if str(e) else repr(e)
                # Si le modèle est en cours de chargement sur HF
                if "loading" in last_error.lower() and attempt < max_retries - 1:
                    time.sleep(10)
                    continue

        raise RuntimeError(f"Erreur ASR (InferenceClient) : {last_error}")

        # Décodage des IDs en texte
        #le son est découpé en des petites briques qui sont transformés par des logits.
        #Pour tout caractère de son découpé on lui donne un score et ce score va correspondre à des lettres
        #ces lettres serviront pour la transcription.
        #En somme, on passe de l'audio au textes. Ces score (valeurs numériques qui seront utilisées dans les calculs
        # pour prédire la traduction de l'audio en textes)
        
        # transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # return transcription.strip()