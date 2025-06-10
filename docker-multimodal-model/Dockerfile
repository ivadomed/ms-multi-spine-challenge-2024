# Use a lightweight Python image
FROM python:3.10-slim

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Creates a directory for the application
WORKDIR /app_challenge

RUN echo "Installing dependencies..."

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

RUN sct_deepseg spinalcord -install

# Upgrade pip 
RUN pip install --upgrade pip

# Clone the nnUNet repository and install the dependencies
RUN git clone https://github.com/plbenveniste/nnUNet.git
RUN git -C nnUNet checkout plb/my_trainers
RUN pip install -e nnUNet/.

# Download the 5 folds of the model
## Create a directory for the trained model
# Create folder structure
RUN for i in 0 1 2 3 4; do \
    mkdir -p nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_${i}; \
done

# Set environment variable to simplify curl commands
ENV BASE_URL=https://github.com/tomDag25/model_repository/releases/download/R1/

# Download the files
RUN wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/dataset.json ${BASE_URL}dataset.json && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/dataset_fingerprint.json ${BASE_URL}dataset_fingerprint.json && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/plans.json ${BASE_URL}plans.json && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_0/checkpoint_best_0.pth ${BASE_URL}checkpoint_best_0.pth && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_1/checkpoint_best_1.pth ${BASE_URL}checkpoint_best_1.pth && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_2/checkpoint_best_2.pth ${BASE_URL}checkpoint_best_2.pth && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_3/checkpoint_best_3.pth ${BASE_URL}checkpoint_best_3.pth && \
    wget -q --show-progress -O nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_4/checkpoint_best_4.pth ${BASE_URL}checkpoint_best_4.pth

# Copy the script files into the container
COPY scripts/ scripts/
# Install Python packages from requirements.txt
RUN pip install -r scripts/requirements.txt

# Set environment variables for nnUNet
ENV nnUNet_raw=nnUNet_datasets \
    nnUNet_results=nnUNet_results \
    nnUNet_preprocessed=nnUNet_preprocessed

# Create directories for nnUNet datasets and results
RUN mkdir -p nnUNet_datasets/Dataset200_MsMultiSpine \
    && mkdir -p nnUNet_preprocessed/Dataset200_MsMultiSpine

# Point d’entrée par défaut (peut être redéfini au runtime)
ENTRYPOINT ["python", "scripts/main.py"]