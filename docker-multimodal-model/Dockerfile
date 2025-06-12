# Dockerfile (Final version)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app_challenge

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

# SCT setup
RUN git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git /root/sct && \
    /root/sct/install_sct -y

ENV PATH="/root/sct/bin:$PATH"
RUN echo "source /root/sct/python/etc/profile.d/conda.sh && conda activate venv_sct" >> /root/.bashrc
RUN sct_deepseg spinalcord -install

# Python dependencies
RUN pip install --upgrade pip

RUN git clone https://github.com/plbenveniste/nnUNet.git && \
    git -C nnUNet checkout plb/my_trainers && \
    pip install -e nnUNet/.

# Store models safely in image
RUN mkdir -p /opt/nnUNet_results/Dataset200_MsMultiSpine
RUN mkdir -p /opt/nnUNet_datasets/Dataset200_MsMultiSpine
RUN mkdir -p /opt/nnUNet_preprocessed/Dataset200_MsMultiSpine


RUN for i in 0 1 2 3 4; do \
    mkdir -p /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_${i}; \
done

ENV BASE_URL=https://github.com/tomDag25/model_repository/releases/download/R1/

RUN wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/dataset.json ${BASE_URL}dataset.json && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/dataset_fingerprint.json ${BASE_URL}dataset_fingerprint.json && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/plans.json ${BASE_URL}plans.json && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_0/checkpoint_best_0.pth ${BASE_URL}checkpoint_best_0.pth && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_1/checkpoint_best_1.pth ${BASE_URL}checkpoint_best_1.pth && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_2/checkpoint_best_2.pth ${BASE_URL}checkpoint_best_2.pth && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_3/checkpoint_best_3.pth ${BASE_URL}checkpoint_best_3.pth && \
    wget -q -O /opt/nnUNet_results/Dataset200_MsMultiSpine/nnUNetTrainerDA5_150epochs__nnUNetResEncUNetLPlans__2d/fold_4/checkpoint_best_4.pth ${BASE_URL}checkpoint_best_4.pth

# Copy and install your scripts
COPY scripts/ scripts/
RUN pip install -r scripts/requirements.txt

# Set nnUNet env
ENV nnUNet_raw=/opt/nnUNet_datasets \
    nnUNet_results=/opt/nnUNet_results \
    nnUNet_preprocessed=/opt/nnUNet_preprocessed


ENTRYPOINT ["python", "scripts/main.py"]
