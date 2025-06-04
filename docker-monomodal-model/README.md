# Docker for monomodal model for MS lesion segmentation

Information on how this was supposed to be built can be found here: https://gitlab.inria.fr/msmultispinechallenge/msmultispinespecification

## Docker installation

Docker is already installed on romane. Installation instructions can be found here: https://docs.docker.com/engine/install/ubuntu/

## Inference scripts

The run_inference_monomodal.py file launches the inference script to segment MS lesions in the spinal cord on the given data. 

Steps: 
1. Data preprocessing
2. Inference on images
3. Fusion of information
4. Calibration
5. Instance segmentation
6. Post-processing for output

The instructions were the following: _your method that can be run with two arguments:_
- _Path to the input directory including the input data for a given case._
- _Path to the output folder that will contain the labeled segmentation mask and associated probability csv._


## Building docker image

To build the docker: 
```console
docker build -t ms-inference-cpu ms-multi-spine-challenge-2024/docker-monomodal-model/.
```

To run the docker:
```console
docker run --rm -v ~/Documents/ms-challenge-2025/sub-001:/input -v ~/Documents/ms-challenge-2025/output_sub001:/output  ms-inference-cpu -i /input -o /output
```

> [!NOTE]  
> the `--rm` flag makes sure that the docker is closed at the end
> the `-v` flag mounts the volume, so that they are visible to the docker


