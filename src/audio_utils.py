import os
from pydub import AudioSegment

def preprocess_audio(input_path: str) -> str:
    """Rééchantillonnage à 16 kHz et conversion en mono."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    output_path = f"processed_{os.path.basename(input_path)}"
    audio.export(output_path, format="wav")
    return output_path

def validate_and_load_audio(input_path: str) -> str:
    """Valide l'existence du fichier et applique le prétraitement."""
    return preprocess_audio(input_path)