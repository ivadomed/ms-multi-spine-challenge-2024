FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone nnUNet
RUN git clone https://github.com/plbenveniste/nnUNet.git

# Install it in editable mode
RUN pip install -e ./nnUNet

RUN git -C nnUNet checkout plb/my_trainers

# Set up the nnUNet variable 
ENV nnUNet_raw=/workspace/nnUNet_datasets
ENV nnUNet_preprocessed=/workspace/nnUNet_preprocessed
ENV nnUNet_results=/workspace/nnUNet_results

# Run requirements.txt
# Copy requirements.txt into the image
COPY requirements.txt /workspace/requirements.txt

# Install it 
RUN pip install -r /workspace/requirements.txt


# Set up the inference command as entrypoint
CMD nnUNetv2_predict \
    -d Dataset170_MsMultiSpine \
    -i /workspace/nnUNet_datasets/Dataset170_MsMultiSpine/imagesTs \
    -o /workspace/result \
    -f 0 \
    -tr nnUNetTrainerDiceCELoss_noSmooth_300epochs \
    -c 2d \
    -p nnUNetResEncUNetLPlans \
    -device cpu\
    -npp 1 \
    -nps 1
