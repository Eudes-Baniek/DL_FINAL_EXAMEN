import os
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

class ASRModel:
    def __init__(self):
        print(f"Chargement du modèle ASR Wav2Vec 2.0 ({MODEL_NAME})...")
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        """Transcrit l'audio 16kHz via Wav2Vec 2.0."""
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError(f"Fichier audio introuvable : {audio_path}")

        try:
            # I. Chargement et rééchantillonnage à 16kHz avec librosa
            speech, sr = librosa.load(audio_path, sr=16000)

            # II. Prétraitement des valeurs d'entrée
            input_values = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_values

            # III. Inférence (sans calcul de gradient)
            
            with torch.no_grad():
                logits = self.model(input_values).logits

            # IV. Décodage des argmax (prédiction des tokens)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription.strip().lower()

        except Exception as e:
            raise RuntimeError(f"Erreur ASR (Wav2Vec 2.0) : {str(e)}")