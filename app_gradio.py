import gradio as gr
from src.pipeline import AudioSentimentPipeline

# Initialisation du pipeline global
pipeline = AudioSentimentPipeline()


def process_audio(audio_path):
    if audio_path is None:
        return "Veuillez fournir un fichier audio valide.", "", ""

    try:
        results = pipeline.predict(audio_path)
        transcription = results["transcription"]
        sentiment = results["sentiment"].capitalize()
        confidence = f"{results['confidence'] * 100:.2f} %"

        return transcription, sentiment, confidence

    except Exception as e:
        return f"Erreur lors du traitement : {str(e)}", "", ""

# Interface Gradio avec import de fichier & enregistrement micro

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(
        type="filepath",
        label="Déposez votre fichier audio (.wav, .mp3, .aac)",
    ),
    outputs=[
        gr.Textbox(label="1. Transcription ASR (Whisper)", lines=4),
        gr.Textbox(
            label="2. Sentiment prédit (CamemBERT / RoBERTa)", lines=1
        ),
        gr.Textbox(label="3. Score de confiance", lines=1),
    ],
    title="🎙️ Détection Automatique de Sentiment dans les Appels Vocaux",
    description=(
        "Ce système transcrit la voix d'un client puis évalue son état"
        " d'esprit (Positif, Négatif, Neutre)."
    ),
)

if __name__ == "__main__":
# Render injecte la variable PORT dynamiquement
    port = int(os.environ.get("PORT", 7860))
    
    # server_name="0.0.0.0" est INDISPENSABLE pour les conteneurs/Render
    demo.launch(server_name="0.0.0.0", server_port=port)
