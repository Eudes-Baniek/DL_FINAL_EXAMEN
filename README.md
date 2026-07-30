# Docker

# Construction de l'image

docker build -t audio-analyse-sentiment-app .

# lancement du container

docker run -p 7860:7860 audio-analyse-sentiment-app

# Versionning avec Git hub

# 1. Initialiser le dépôt Git

git init

# 2. Définir la branche principale sur 'main'

git branch -M main

# 3. Ajouter tous les fichiers (Git va ignorer ce qui est spécifié dans .gitignore)

git add .

# 4. Créer votre premier point de sauvegarde (Commit)

git commit -m "Initial commit - Pipeline Audio, Dockerfile et Gradio"
