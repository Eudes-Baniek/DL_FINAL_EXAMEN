import os
import shutil
import numpy as np
from pydub import AudioSegment

# Chemin exact trouvé (FFmpeg 8.1.2) sur la machine 

# Détection automatique de FFmpeg dans le système (Docker et Serveurs Linux)

FFMPEG_BIN = shutil.which("ffmpeg")

#Si non trouvé dans le PATH, on utilise le fallback Windows

if not FFMPEG_BIN:
    FFMPEG_BIN = r"C:\Users\Best Computer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"

#Configuration de PyDub et ajout au PATH si le fichier existe

if os.path.exists(FFMPEG_BIN):
    AudioSegment.converter = FFMPEG_BIN
    FFMPEG_DIR = os.path.dirname(FFMPEG_BIN)
    if FFMPEG_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# FFMPEG_BIN = r"C:\Users\Best Computer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
# FFMPEG_DIR = os.path.dirname(FFMPEG_BIN)
# On indique le chemin exact à PyDub et ajoute le dossier au PATH

AudioSegment.converter = FFMPEG_BIN
if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR

MAX_DURATION_SECONDS = 300  # 5 minutes maximum
TARGET_SAMPLING_RATE = 16000  # 16 kHz requis pour Wav2Vec 2.0

#Liste élargie à TOUS les formats audio courants
#pour que le projet soit plus pratique et vu que j'utilise les audio des
#conversation whatsap et autres pour faire les tests j'ai mis plusieurs 
#extensions

SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".aac",
    ".acc",
    ".m4a",
    ".flac",
    ".ogg",
    ".wma",
    ".opus",
    ".amr",
    ".aiff",
    ".webm",
)

def validate_and_load_audio(file_path: str):
    # Valide et charge un fichier audio
    # Convertit en mono
    # Rééchantillonne à 16 kHz
    # Vérifie la durée (< 5 min)
    # Gère les fichiers vides/silencieux


    # 1. Vérification de l'existence du fichier
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # 2. Validation de l'extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format '{ext}' non supporté. Formats acceptés :"
            f" {SUPPORTED_EXTENSIONS}"
        )

    # 3. Vérification de la taille (non vide)
    if os.path.getsize(file_path) == 0:
        raise ValueError("Le fichier audio est vide.")

    # 4. Chargement, conversion en mono et rééchantillonnage
    try:
        # PyDub gère le décodage de tous ces formats grâce à FFmpeg
        
        segment = AudioSegment.from_file(file_path)

        # Passage en mono
        segment = segment.set_channels(1)

        # Rééchantillonnage à 16 kHz
        
        segment = segment.set_frame_rate(TARGET_SAMPLING_RATE)

        # Conversion en numpy array float32 (normalisé entre -1.0 et 1.0)
        
        audio = np.array(segment.get_array_of_samples()).astype(np.float32)

        if segment.sample_width == 2:
            audio /= 32768.0
        elif segment.sample_width == 4:
            audio /= 2147483648.0

        sr = TARGET_SAMPLING_RATE

    except Exception as e:
        raise RuntimeError(
            f"Erreur lors de la lecture du fichier audio ({ext}) : {str(e)}"
        )

    # 5. Vérification du silence
    if len(audio) == 0 or np.all(np.abs(audio) < 1e-4):
        raise ValueError(
            "Le fichier audio est silencieux ou ne contient aucun signal"
            " exploitable."
        )

    # 6. Vérification de la durée max
    duration = len(audio) / sr
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"La durée de l'audio ({duration:.1f}s) dépasse la limite autorisée"
            f" de {MAX_DURATION_SECONDS}s."
        )

    return audio, sr