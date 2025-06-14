# Use a lightweight Python image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Creates a directory for the application
WORKDIR /app_challenge

# Install Python dependencies
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

# Install SCT
RUN git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git /root/sct && \
    /root/sct/install_sct -y
# Add SCT to PATH
ENV PATH="/root/sct/bin:$PATH"
# Automatically activate SCT env in bash
RUN echo "source /root/sct/python/etc/profile.d/conda.sh && conda activate venv_sct" >> /root/.bashrc
RUN sct_deepseg spinalcord -install

# Python dependencies
RUN pip install --upgrade pip

# Clone the nnUNet repository and install the dependencies
RUN git clone https://github.com/plbenveniste/nnUNet.git
RUN git -C nnUNet checkout plb/my_trainers
RUN pip install -e nnUNet/.

# Store models safely in image
RUN mkdir -p /opt/nnUNet_results/Dataset151_MsMultiSpine
RUN mkdir -p /opt/nnUNet_datasets/Dataset151_MsMultiSpine
RUN mkdir -p /opt/nnUNet_preprocessed/Dataset151_MsMultiSpine

RUN for i in 0 1 2 3 4; do \
    mkdir -p /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_${i}; \
done

ENV BASE_URL=https://github.com/plbenveniste/model_repository/releases/download/R1/

RUN wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/dataset.json ${BASE_URL}dataset.json && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/dataset_fingerprint.json ${BASE_URL}dataset_fingerprint.json && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/plans.json ${BASE_URL}plans.json && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_0/checkpoint_best.pth ${BASE_URL}checkpoint_best_0.pth && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_1/checkpoint_best.pth ${BASE_URL}checkpoint_best_1.pth && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_2/checkpoint_best.pth ${BASE_URL}checkpoint_best_2.pth && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_3/checkpoint_best.pth ${BASE_URL}checkpoint_best_3.pth && \
    wget -q -O /opt/nnUNet_results/Dataset151_MsMultiSpine/nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch__nnUNetResEncUNetLPlansFinetune__3d_fullres/fold_4/checkpoint_best.pth ${BASE_URL}checkpoint_best_4.pth

# Copy the script files into the container
COPY scripts/ scripts/

# Install Python packages from requirements.txt
RUN pip install -r scripts/requirements.txt

# Set nnUNet env
ENV nnUNet_raw=/opt/nnUNet_datasets \
    nnUNet_results=/opt/nnUNet_results \
    nnUNet_preprocessed=/opt/nnUNet_preprocessed

# Point d’entrée par défaut (peut être redéfini au runtime)
ENTRYPOINT ["python", "scripts/main.py"]
