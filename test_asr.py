import sys
import os
# from .audio_utils import validate_and_load_audio
# from .asr_model import ASRModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio_utils import validate_and_load_audio
from src.asr_model import ASRModel

def main():
    # Remarque le 'r' devant les guillemets pour annuler les problèmes d'anti-slash Windows
    audio_path = r"C:\DL_FINAL_EXAMEN\data\AUD-20260722-WA0004.aac"
    
    print("--- 1. Chargement et prétraitement de l'audio ---")
    try:
        audio, sr = validate_and_load_audio(audio_path)
        print(f" Audio chargé avec succès ! (Échantillonnage : {sr} Hz, Taille : {audio.shape})")
    except Exception as e:
        print(f" Erreur lors du chargement de l'audio : {e}")
        return

    print("\n--- 2. Chargement du modèle Wav2Vec 2.0 ---")
    try:
        asr = ASRModel()
        print(f" Modèle ASR chargé sur l'appareil : {asr.device}")
    except Exception as e:
        print(f" Erreur lors du chargement du modèle : {e}")
        return

    print("\n--- 3. Transcription en cours... ---")
    try:
        transcription = asr.transcribe(audio, sampling_rate=sr)
        print("\n" + "="*50)
        print("TRANSCRIPTION OBTENUE :")
        print(f"\"{transcription}\"")
        print("="*50)
    except Exception as e:
        print(f" Erreur pendant la transcription : {e}")

if __name__ == "__main__":
    main()          