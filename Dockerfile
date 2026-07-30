# I. Image de base Python 3.10 slim
FROM python:3.10-slim

# Empêche la génération des fichiers .pyc et assure la diffusion en direct des logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# II. Installation de FFmpeg et des bibliothèques système nécessaires pour l'audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# III. Répertoire de travail du conteneur
WORKDIR /app

# IV. Copie et installation des dépendances Python
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# V. Copie du reste du code source du projet
COPY . .

# VI. Variables d'environnement pour Gradio (écoute globale pour conteneur/cloud)
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Port exposé par Gradio
EXPOSE 7860

# VII. Lancement de l'application
CMD ["python", "app_gradio.py"]