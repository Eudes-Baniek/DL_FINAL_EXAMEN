# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux

Projet d'Examen — **Deep Learning 2 (2026)**  
**Établissement :** DIT (Dakar Institute of Technology)  
**Auteur :** Eudes Exaucé Baniek

---

## 1. Présentation du Projet

Dans le cadre des centres d'appels clients, l'analyse manuelle des enregistrements vocaux est coûteuse et chronophage. Ce projet propose un pipeline automatisé capable de :

1. **Transcrire** un fichier audio français (`.wav`, `.mp3`) en texte via de la Reconnaissance Automatique de la Parole (ASR).
2. **Analyser le sentiment** du texte transcrit pour classifier l'appel en 3 catégories : **Positif**, **Neutre** ou **Négatif**, accompagné d'un score de confiance.

---

## 2. Architecture & Workflow

## Modèles Utilisés

# ASR (Speech-to-Text) : jonatasgrosman/wav2vec2-large-xlsr-53-french

Analyse de Sentiment (NLP) : Modèle basé sur l'architecture CamemBERT affiné pour la classification de texte en français.

Stack Technique

Langage : Python 3.10

Framework Web & UI : Gradio, FastAPI, Uvicorn

Deep Learning & Audio : PyTorch (CPU), Transformers (Hugging Face), Librosa, SoundFile, Pydub

Conteneurisation & CI/CD : Docker, Render Cloud Services

## 4. Structure du Dépôt

```bash
.
├── src/
│   ├── asr_model.py       # Module de transcription audio (Groq Client)
│   ├── sentiment_model.py # Module d'analyse de sentiment (HF API)
│   └── pipeline.py        # Chef d'orchestre reliant ASR -> NLP
├── data/               # Audios d'exemple pour les tests (Positif, Neutre, Négatif)
├── app_gradio.py          # Interface Utilisateur Gradio
├── Dockerfile             # Configuration pour conteneurisation Docker
├── requirements.txt       # Dépendances Python du projet
└── README.md
└── Images                 #Quelques captures d'écran
            # Documentation du projet
5. Installation & Lancement Local

Étape 1 : Cloner le dépôt et installer les dépendances

git clone https://github.com/Eudes-Baniek/DL_FINAL_EXAMEN.git

Étape 2 : Lancer l'interface Gradio

python app_gradio.py ou py python app_gradio.py

L'interface sera accessible sur http://localhost:7860.

6. Lancement avec Docker


a. Construction de l'image

docker build -t audio-analyse-sentiment-app .

b. lancement du container

docker run -p 7860:7860 audio-analyse-sentiment-app


# Cette partie : Elle concerne le déploiement hors projet que j'ai eu à faire

   NB: Etant confronté au problemes de Memoire (512Mo de RAM maxi sur Render ) sur Render pour le déploiement,
   pour etre conformes aux instructions du projet, j'ai décidé de ne pas faire le délpoiement avec le modele
   Wav2Vec2.0. Cependant, pour ma propre gouverne, j'après expérienté le déploiement en utilisant Groq (jeton)

Audio (.wav / .mp3)
│
▼
[ Pipeline ASR ] ──► Groq API (Whisper-Large-v3-Turbo)
│
▼
Transcription Texte
│
▼
[ Pipeline NLP ] ──► Hugging Face API (DistilCamemBERT)
│
▼
Résultat Final : Sentiment (Positif/Neutre/Négatif) + Confiance (%)

**Déploiement en ligne :** https://dl-final-examen.onrender.com/

Pour garder les trace de ce que j'avais fait, j'ai laissé les captures d'écran qui atteste du déploiement fait
avec le modele 'Whisper-Large-v3-Turbo' et qui a bien marché.

```
