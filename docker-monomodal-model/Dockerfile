# Use a lightweight Python image
FROM python:3.10-slim

# Creates a directory for the application
WORKDIR /app_challenge

# Copy the script files into the container (they are not anymore in a scripts folder)
COPY scripts/ scripts/

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install SCT
RUN git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git
RUN spinalcordtoolbox/install_sct -y

# Install Python packages from requirements.txt
RUN pip install -r scripts/requirements.txt

# Clone the nnUNet repository and install the dependencies
RUN git clone https://github.com/plbenveniste/nnUNet.git
RUN git -C nnUNet checkout plb/my_trainers
RUN pip install -e nnUNet/.

# Download the 5 folds of the model
## Create a directory for the trained model
RUN mkdir -p trained-model
## Downloading the model
RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold0.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold1.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold2.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold3.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold4.zip
## Unzip the model files into the trained-model directory
RUN unzip model_fold0.zip -d trained-model/
# RUN unzip model_fold1.zip -d trained-model/
# RUN unzip model_fold2.zip -d trained-model/
# RUN unzip model_fold3.zip -d trained-model/
# RUN unzip model_fold4.zip -d trained-model/

# Point d’entrée par défaut (peut être redéfini au runtime)
ENTRYPOINT ["python", "scripts/main.py"]
