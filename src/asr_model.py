import os
import time
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

class ASRModel:
    def __init__(self):
        self.client = InferenceClient(
            model="openai/whisper-small",
            token=HF_TOKEN
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcrit l'audio via le client Inference officiel Hugging Face."""
        if not audio_path or not os.path.exists(audio_path):
            return ""

        # Tentatives avec gestion du démarrage à chaud du modèle HF
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.automatic_speech_recognition(audio_path)
                
                if isinstance(result, dict):
                    return result.get("text", "").strip()
                elif hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()

            except Exception as e:
                err_msg = str(e)
                # Si le modèle est en train de charger sur les serveurs HF
                if "loading" in err_msg.lower() and attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                raise RuntimeError(f"Erreur ASR (InferenceClient) : {err_msg}")

        # Décodage des IDs en texte
        #le son est découpé en des petites briques qui sont transformés par des logits.
        #Pour tout caractère de son découpé on lui donne un score et ce score va correspondre à des lettres
        #ces lettres serviront pour la transcription.
        #En somme, on passe de l'audio au textes. Ces score (valeurs numériques qui seront utilisées dans les calculs
        # pour prédire la traduction de l'audio en textes)
        
        # transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # return transcription.strip()