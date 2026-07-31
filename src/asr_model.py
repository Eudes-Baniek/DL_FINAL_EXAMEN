import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
ASR_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-small"

class ASRModel:
    def transcribe(self, audio_path: str) -> str:
        # Transcrit l'audio via l'API Inference de Hugging Face (Whisper)
        if not audio_path or not os.path.exists(audio_path):
            return ""

        with open(audio_path, "rb") as f:
            data = f.read()

        response = requests.post(ASR_URL, headers=HEADERS, data=data, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"Erreur ASR (Code {response.status_code}) : {response.text}")

        result = response.json()
        if isinstance(result, dict):
            return result.get("text", "").strip()
        elif isinstance(result, list) and len(result) > 0:
            return result[0].get("text", "").strip()
        return str(result).strip()

        # Décodage des IDs en texte
        #le son est découpé en des petites briques qui sont transformés par des logits.
        #Pour tout caractère de son découpé on lui donne un score et ce score va correspondre à des lettres
        #ces lettres serviront pour la transcription.
        #En somme, on passe de l'audio au textes. Ces score (valeurs numériques qui seront utilisées dans les calculs
        # pour prédire la traduction de l'audio en textes)
        
        # transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # return transcription.strip()