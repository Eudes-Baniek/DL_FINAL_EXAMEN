# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux

Projet d'Examen — **Deep Learning 2 (2026)**  
**Établissement :** DIT (Dakar Institute of Technology)  
**Auteur :** Eudes Exaucé Baniek  
**Déploiement en ligne :** https://dl-final-examen.onrender.com/

---

## 1. Présentation du Projet

Dans le cadre des centres d'appels clients, l'analyse manuelle des enregistrements vocaux est coûteuse et chronophage. Ce projet propose un pipeline automatisé capable de :

1. **Transcrire** un fichier audio français (`.wav`, `.mp3`) en texte via de la Reconnaissance Automatique de la Parole (ASR).
2. **Analyser le sentiment** du texte transcrit pour classifier l'appel en 3 catégories : **Positif**, **Neutre** ou **Négatif**, accompagné d'un score de confiance.

---

## 2. Architecture & Workflow

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

## 3. Choix des Modèles & Justification

| Tâche                    | Modèle / Solution                                                     | Justification du Choix                                                                                                                                                                         |
| :----------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ASR (Speech-to-Text)** | `openai/whisper-large-v3-turbo` (via Groq API)                        | Modèle état de l'art pour la reconnaissance vocale française. L'exécution via les LPU de Groq offre un temps de réponse sub-secondaire et résout la contrainte de mémoire RAM sur l'hébergeur. |
| **Sentiment (NLP)**      | `cmarkea/distilcamembert-base-sentiment` (via Hugging Face Inference) | Modèle BERT compact spécifiquement entraîné et ajusté pour la classification de sentiment sur le français (CamemBERT).                                                                         |

---

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

Clé API Groq (stockée dans HF_TOKEN:votre_cle_groq_ici)

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

```
