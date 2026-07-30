import json
import sys
from src.pipeline import AudioSentimentPipeline

# Chemin du fichier audio à tester
AUDIO_PATH = r"C:\DL_FINAL_EXAMEN\data\AUD-20260705-WA0003.aac"

print("==*10")
print(" 🎙️ PIPELINE D'ANALYSE DE SENTIMENT VOCALE")
print("==*10\n")

try:
    print("1. Initialisation des modèles ML...")
    pipeline = AudioSentimentPipeline()

    print(f"\n2. Traitement du fichier audio : {AUDIO_PATH}")
    result = pipeline.predict(AUDIO_PATH)

    print("\n==*10")
    print(" RESULTATS OBTENUS :")
    print("==*10")
    print(f"• Transcription ASR : {result['transcription']}")
    print(f"• Sentiment Prédit : {result['sentiment'].upper()}")
    print(f"• Score de Confiance : {result['confidence'] * 100:.2f}%")
    print("==*10")

    print("\nFormat JSON de sortie :")
    print(json.dumps(result, ensure_ascii=False, indent=2))

except Exception as e:
    print(f"\n❌ Erreur lors de l'exécution du pipeline : {e}")
    sys.exit(1) 