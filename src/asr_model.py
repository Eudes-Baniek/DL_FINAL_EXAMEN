import os
from groq import Groq

# Récupération de la clé Groq stockée dans la variable HF_TOKEN sur Render
API_KEY = os.getenv("HF_TOKEN")

class ASRModel:
    def __init__(self):
        if not API_KEY:
            raise ValueError("La variable d'environnement HF_TOKEN n'est pas définie sur Render.")
        self.client = Groq(api_key=API_KEY)

    def transcribe(self, audio_path: str) -> str:
        """Transcrit l'audio via Whisper Large v3 chez Groq."""
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Fichier audio introuvable ou chemin invalide.")

        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3-turbo",
                    language="fr",
                    response_format="text"
                )
            
            return transcription.strip() if isinstance(transcription, str) else str(transcription).strip()

        except Exception as e:
            raise RuntimeError(f"Erreur ASR (Groq Whisper) : {str(e)}")