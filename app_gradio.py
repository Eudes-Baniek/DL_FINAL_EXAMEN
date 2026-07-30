# import gradio as gr
# from src.pipeline import AudioSentimentPipeline

# # Initialisation du pipeline global
# pipeline = AudioSentimentPipeline()


# def process_audio(audio_path):
#     if audio_path is None:
#         return "Veuillez fournir un fichier audio valide.", "", ""

#     try:
#         results = pipeline.predict(audio_path)
#         transcription = results["transcription"]
#         sentiment = results["sentiment"].capitalize()
#         confidence = f"{results['confidence'] * 100:.2f} %"

#         return transcription, sentiment, confidence

#     except Exception as e:
#         return f"Erreur lors du traitement : {str(e)}", "", ""

# # Interface Gradio avec import de fichier & enregistrement micro

# demo = gr.Interface(
#     fn=process_audio,
#     inputs=gr.Audio(
#         type="filepath",
#         label="Déposez votre fichier audio (.wav, .mp3, .aac)",
#     ),
#     outputs=[
#         gr.Textbox(label="1. Transcription ASR (Wav2Vec 2.0)", lines=4),
#         gr.Textbox(
#             label="2. Sentiment prédit (CamemBERT / RoBERTa)", lines=1
#         ),
#         gr.Textbox(label="3. Score de confiance", lines=1),
#     ],
#     title="🎙️ Détection Automatique de Sentiment dans les Appels Vocaux",
#     description=(
#         "Ce système transcrit la voix d'un client puis évalue son état"
#         " d'esprit (Positif, Négatif, Neutre)."
#     ),
# )

# if __name__ == "__main__":
#     demo.launch()



import os
import gradio as gr
from huggingface_hub import InferenceClient

# Jeton récupéré depuis les variables d'environnement de Render

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialisation du client d'inférence très léger

client = InferenceClient(token=HF_TOKEN)

ASR_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
SENTIMENT_MODEL = "cmarkea/distilcamembert-base-sentiment"

def process_audio(audio_path):
    if not audio_path:
        return "Aucun fichier audio fourni.", ""

    try:
        # 1. Envoi de l'audio à l'Inference API HF pour transcription
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        asr_response = client.automatic_speech_recognition(
            audio_bytes, 
            model=ASR_MODEL
        )
        transcription = asr_response.get("text", "")

        if not transcription.strip():
            return "Transcription vide ou audio silencieux.", "N/A"

        # 2. Envoi du texte à l'API HF pour l'analyse de sentiment
        
        sentiment_response = client.text_classification(
            transcription, 
            model=SENTIMENT_MODEL
        )
        
        # Formattage des résultats
        
        results = [f"{item.get('label', 'Inconnu')}: {item.get('score', 0.0):.2%}" for item in sentiment_response]
        return transcription, "\n".join(results)

    except Exception as e:
        return f"Erreur lors du traitement : {str(e)}", "Erreur"

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Fichier audio (.wav / .mp3)"),
    outputs=[
        gr.Textbox(label="Transcription (Wav2Vec 2.0)"),
        gr.Textbox(label="Sentiment (CamemBERT)")
    ],
    title="Détection de Sentiment dans les Appels Vocaux",
    description="Pipeline ASR -> NLP déployé via Hugging Face Inference API"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)