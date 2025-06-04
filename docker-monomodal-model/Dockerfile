# Use a lightweight Python image
FROM python:3.10-slim

# Creates a directory for the application
WORKDIR /app_challenge

# Copy the script files into the container (they are not anymore in a scripts folder)
COPY scripts/ scripts/
COPY trained-model/ trained-model/

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from requirements.txt
RUN pip install -r scripts/requirements.txt

# Clone the nnUNet repository and install the dependencies
RUN git clone https://github.com/plbenveniste/nnUNet.git
RUN git -C nnUNet checkout plb/my_trainers
RUN pip install -e nnUNet/.

# Point d’entrée par défaut (peut être redéfini au runtime)
ENTRYPOINT ["python", "scripts/main.py"]
