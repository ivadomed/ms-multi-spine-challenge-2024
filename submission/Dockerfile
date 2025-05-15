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
ENV nnUNet_raw=/workspace
ENV nnUNet_preprocessed=/workspace/nnUNet_preprocessed
ENV nnUNet_results=/workspace/nnUNet_results


# Run requirements.txt
# Copy requirements.txt into the image
COPY requirements.txt /workspace/requirements.txt

# Install it 
RUN pip install -r /workspace/requirements.txt


# Set up the inference command as entrypoint
CMD bash -c "\
    for case in \$(ls /workspace/Dataset170_MsMultiSpine/imagesTs/); do \
        case_id=\$(basename \$case); \
        echo Predicting \$case_id && \
        nnUNetv2_predict \
            -i /workspace/Dataset170_MsMultiSpine/imagesTs/\$case_id \
            -o /workspace/result \
            -d Dataset170_MsMultiSpine \
            -tr nnUNetTrainerDiceCELoss_noSmooth_300epochs \
            -c 2d \
            -p nnUNetResEncUNetLPlans \
            -device cpu \
            -f 0; \
    done"
