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
        """Transcrit l'audio via l'InferenceClient d'Hugging Face en gérant le flux/générateur."""
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Fichier audio introuvable ou chemin invalide.")

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le fichier audio : {str(e)}")

        max_retries = 3
        last_error = ""

        for attempt in range(max_retries):
            try:
                # Appels de reconnaissance vocale via le client HF
                response = self.client.automatic_speech_recognition(audio_bytes)
                
                # 1. Si response est un générateur / iterateur (cause du StopIteration)
                if hasattr(response, "__iter__") and not isinstance(response, (dict, str, list)):
                    full_text = []
                    for chunk in response:
                        if isinstance(chunk, dict):
                            full_text.append(chunk.get("text", ""))
                        elif hasattr(chunk, "text"):
                            full_text.append(chunk.text)
                        else:
                            full_text.append(str(chunk))
                    return "".join(full_text).strip()

                # 2. Si response est un dictionnaire direct {"text": "..."}
                if isinstance(response, dict):
                    return response.get("text", "").strip()

                # 3. Si response a un attribut .text
                if hasattr(response, "text"):
                    return response.text.strip()

                return str(response).strip()

            except Exception as e:
                # Évite d'attraper le StopIteration pour rien si c'était lié à une mauvaise itération
                last_error = str(e) if str(e) else repr(e)
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