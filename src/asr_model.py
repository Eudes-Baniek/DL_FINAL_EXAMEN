import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# On a pris le modèle recommandé dans le projet

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

class ASRModel:
    def __init__(self, model_name: str = MODEL_NAME):

        # Initialisation du le processeur et du modèle Wav2Vec 2.0.
        #Et on utilise le GPU s'est elligible sinon on utilise le cpu

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Chargement du processor (feature extractor + tokenizer)
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Chargement du modèle de reconnaissance vocale
        
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Mode évaluation (désactive le dropout)
        
        self.model.eval()  

    def transcribe(self, audio_array, sampling_rate: int = 16000) -> str:
    
        # Transcription des matrices en texte
        # Prend un tableau audio (NumPy) à 16kHz et retourne la transcription texte.
      
        if len(audio_array) == 0:
            return ""

        # Pre-processing de l'audio pour le modèle
        
        inputs = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        # Inférence (sans calcul de gradients pour économiser la mémoire)
        
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Prédiction des IDs des tokens les plus probables
        
        predicted_ids = torch.argmax(logits, dim=-1)

        # Décodage des IDs en texte
        #le son est découpé en des petites briques qui sont transformés par des logits.
        #Pour tout caractère de son découpé on lui donne un score et ce score va correspondre à des lettres
        #ces lettres serviront pour la transcription.
        #En somme, on passe de l'audio au textes. Ces score (valeurs numériques qui seront utilisées dans les calculs
        # pour prédire la traduction de l'audio en textes)
        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.strip()