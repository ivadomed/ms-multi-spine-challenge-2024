# Use a lightweight Python image
FROM python:3.10-slim

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Creates a directory for the application
WORKDIR /app_challenge

# Install system dependencies needed for SCT and PyQt5
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    bash \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    libglib2.0-dev \
    libgl1-mesa-dev \
    ca-certificates \
    bzip2 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone SCT and install
RUN git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git /root/sct && \
    /root/sct/install_sct -y

# Add SCT to PATH
ENV PATH="/root/sct/bin:$PATH"

# Automatically activate SCT env in bash
RUN echo "source /root/sct/python/etc/profile.d/conda.sh && conda activate venv_sct" >> /root/.bashrc

# Upgrade pip 
RUN pip install --upgrade pip

# Clone the nnUNet repository and install the dependencies
RUN git clone https://github.com/plbenveniste/nnUNet.git
RUN git -C nnUNet checkout plb/my_trainers
RUN pip install -e nnUNet/.

# Download the 5 folds of the model
## Create a directory for the trained model
#RUN mkdir -p trained-model
## Downloading the model
#RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold0.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold1.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold2.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold3.zip
# RUN wget https://github.com/plbenveniste/model_repository/releases/download/R1/model_fold4.zip
## Unzip the model files into the trained-model directory
#RUN unzip model_fold0.zip -d trained-model/
# RUN unzip model_fold1.zip -d trained-model/
# RUN unzip model_fold2.zip -d trained-model/
# RUN unzip model_fold3.zip -d trained-model/
# RUN unzip model_fold4.zip -d trained-model/

# Copy the script files into the container
COPY scripts/ scripts/
# Install Python packages from requirements.txt
RUN pip install -r scripts/requirements.txt

# Point d’entrée par défaut (peut être redéfini au runtime)
ENTRYPOINT ["python", "scripts/main.py"]