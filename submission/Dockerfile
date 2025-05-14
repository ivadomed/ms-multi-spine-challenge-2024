FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# Set work directory
WORKDIR /workspace

# Install basic utilities and git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone your fork of nnUNet
RUN git clone https://github.com/plbenveniste/nnUNet.git

# Install nnUNet in editable mode
RUN pip install -e ./nnUNet

# Set default command to bash (optional: makes it easier to drop in)
CMD ["/bin/bash"]
