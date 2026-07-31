import os
from pydub import AudioSegment

def preprocess_audio(input_path: str) -> str:
    """Rééchantillonnage 16 kHz et conversion Mono."""
    if not os.path.exists(input_path):
        return input_path

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        output_path = f"processed_{os.path.basename(input_path)}"
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"[Warning Preprocessing] : {e}")
        return input_path