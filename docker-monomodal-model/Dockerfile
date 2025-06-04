# Utilise une image officielle légère avec Python préinstallé
FROM python:3.10-slim

# Empêche les prompts interactifs durant les builds
ENV DEBIAN_FRONTEND=noninteractive

# Crée un dossier de travail
WORKDIR /app_challenge

# Copie le fichier d'inférence et les fichiers modules nécessaires dans le conteneur
COPY scripts/ .

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Point d’entrée par défaut (peut être redéfini au runtime)
CMD ["python", "main.py"]
