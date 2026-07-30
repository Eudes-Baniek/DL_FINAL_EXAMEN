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
import requests
import gradio as gr

# Token HF depuis Render
render-deploy = os.getenv("render-deploy")

# En-têtes pour les requêtes HTTP directes
HEADERS = {"Authorization": f"Bearer {render-deploy}"}

# Endpoints Inference API
ASR_URL = "https://router.huggingface.co/hf-inference/models/jonatasgrosman/wav2vec2-large-xlsr-53-french"
SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/cmarkea/distilcamembert-base-sentiment"

def process_audio(audio_path):
    if not audio_path:
        return "Aucun fichier audio fourni.", "N/A"

    if not render-deploy:
        return "ERREUR : La variable d'environnement render-deploy n'est pas définie sur Render !", "Erreur Token"

    try:
        # 1. Lecture de l'audio
        with open(audio_path, "rb") as f:
            data = f.read()

        # 2. Appel API ASR (Transcription)
        response_asr = requests.post(ASR_URL, headers=HEADERS, data=data, timeout=30)
        
        if response_asr.status_code == 503:
            return "Le modèle Wav2Vec2 est en cours de chargement sur Hugging Face (503). Patientez 20 secondes et réessayez.", "Modèle en chauffe"
        
        if response_asr.status_code != 200:
            return f"Erreur API ASR (Code {response_asr.status_code}) : {response_asr.text}", "Erreur ASR"

        result_asr = response_asr.json()
        
        # Extraction du texte transcrit
        if isinstance(result_asr, dict):
            transcription = result_asr.get("text", "")
        elif isinstance(result_asr, list) and len(result_asr) > 0:
            transcription = result_asr[0].get("text", "")
        else:
            transcription = str(result_asr)

        if not transcription.strip():
            return "Transcription vide ou audio inaudible.", "Neutre"

        # 3. Appel API Sentiment
        payload_sentiment = {"inputs": transcription}
        response_sent = requests.post(SENTIMENT_URL, headers=HEADERS, json=payload_sentiment, timeout=20)

        if response_sent.status_code != 200:
            return transcription, f"Erreur API Sentiment (Code {response_sent.status_code}) : {response_sent.text}"

        result_sent = response_sent.json()

        # Formatage des scores de sentiment
        if isinstance(result_sent, list) and len(result_sent) > 0:
            predictions = result_sent[0] if isinstance(result_sent[0], list) else result_sent
            formatted = [f"{item.get('label', 'Inconnu')}: {item.get('score', 0.0):.2%}" for item in predictions]
            sentiment_text = "\n".join(formatted)
        else:
            sentiment_text = str(result_sent)

        return transcription, sentiment_text

    except requests.exceptions.Timeout:
        return "Erreur : Temps d'attente dépassé (Timeout API Hugging Face). Réessayez.", "Timeout"
    except Exception as e:
        return f"Erreur système : {str(e)}", "Erreur"

# Interface Gradio
demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Fichier audio (.wav / .mp3)"),
    outputs=[
        gr.Textbox(label="Transcription (Wav2Vec 2.0)"),
        gr.Textbox(label="Sentiment (CamemBERT)")
    ],
    title="Détection de Sentiment dans les Appels Vocaux",
    description="Pipeline ASR -> NLP via Hugging Face Inference API"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
