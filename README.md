# ms-spine-challenge
Deep learning models for the MS-Multi-Spine challenge 2024: https://portal.fli-iam.irisa.fr/ms-multi-spine/

More information can be found in the Zenodo record: https://zenodo.org/records/14051168


# Docker 


## Basics: 


To install docker you can use the procedure described [here](https://docs.docker.com/engine/install/ubuntu/) (**Not needed on Romane**)

To build a docker image we first need a docker image to start with. 
Since nnUNet uses pythorch with `torch==2.5.1` I will use `docker pull pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel`
The `devel` stands for developpement and is bigger than the `runtime` but it's more complete and allows to run everything you want in pytorch. 

Now to see if everything works fine you can go in the `docker_test` folder and run the command: 

```bash 
docker build -t torch-cuda-test .
docker run --rm --gpus all torch-cuda-test
```

The goal of this test is to check whether you have GPU access or not. You shouldn't have GPU access and get this kind of output on romane. 
```bash 


==========
== CUDA ==
==========

CUDA Version 11.8.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

CUDA available: False
Number of GPUs: 0
```

To note in order to not have containers running in the wild: 
- The flag `--rm` in the `run`command makes sure that after running the docker container is stopped
- If you don't use this flag the container will keep running, to see all running container use `docker ps` 
- If you exited the docker container use `docker ps -a`
- To stop a running container use `docker stop <container-id-or-name>`
- To remove an exited Container use `docker rm <container_id_or_name>`
- You can clean all stopped container with `docker container prune`
- If you want to start an interactive shell in the container use `docker run --rm -it --gpus all torch-cuda-test bash` in our case 
- To exit it use `exit`
- To reenter it use `docker start  <container_id_or_name>`

To make some basic testing use the --rm flag to make sure things don't pile up. 

## For the submission

If you now go in the submission folder you can see a Dockerfile that we will use for submission to the competition. 
Let's break it down: 

Init of the Docker with the torch image
```Python 
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel`

# Set work directory
WORKDIR /workspace
```
Install git and clone our version from the nnUNet github
```Python
# Install basic utilities and git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone your fork of nnUNet
RUN git clone https://github.com/plbenveniste/nnUNet.git

# Install nnUNet in editable mode
RUN pip install -e ./nnUNet

#move to the branch where our trainers are 
RUN git -C nnUNet checkout plb/my_trainers
```

To see if it works you can type: `docker run --rm -it --gpus all torch-nnunet` which should open a docker bash session and type `nnUNetv2_train -h`. 

Then we have to set nnUNet variables: 

```Python
# Set up the nnUNet variable 
ENV nnUNet_raw=/workspace/nnUNet_datasets
ENV nnUNet_preprocessed=/workspace/nnUNet_preprocessed
ENV nnUNet_results=/workspace/nnUNet_results
```
Install the requirements that were not contained in nnUNet.  


```Python 
# Run requirements.txt
# Copy requirements.txt into the image
COPY requirements.txt /workspace/requirements.txt

# Install it 
RUN pip install -r /workspace/requirements.txt
```

Make the nnUNet inference: 

```Python 
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

```

I had to put the npp and nps flags so that nnUNet wouldn't take too much RAM and make the process crash. It is really demanding in CPU mode and it could be an issue for submission. 