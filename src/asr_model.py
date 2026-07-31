import os
import time
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
ASR_URL = "https://router.huggingface.co/hf-inference/models/facebook/wav2vec2-large-xlsr-53-french"

class ASRModel:
    def transcribe(self, audio_path: str) -> str:
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Fichier audio introuvable ou chemin invalide.")

        with open(audio_path, "rb") as f:
            data = f.read()

        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(ASR_URL, headers=HEADERS, data=data, timeout=30)
            
            # Si le modèle est en phase d'échauffement sur les serveurs HF
            if response.status_code == 503 and attempt < max_retries - 1:
                time.sleep(10)
                continue

            if response.status_code != 200:
                raise RuntimeError(f"Code {response.status_code} - {response.text}")

            result = response.json()
            if isinstance(result, dict):
                return result.get("text", "").strip()
            elif isinstance(result, list) and len(result) > 0:
                return result[0].get("text", "").strip()

            return str(result).strip()

        return ""