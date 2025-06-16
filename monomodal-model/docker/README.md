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
docker build -t plbenveniste/ms-challenge-monomodal:2.0 ms-multi-spine-challenge-2024/monomodal-model/docker.
```

> [!NOTE]  
> The `--platform linux/amd64` was added to avoid this [issue](https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4918#issue-3118276041). This was specific to me running it on MacOs.

To run the docker:
```console
docker run --rm -it  -v /home/ge.polymtl.ca/thdaga/docker_pl/sub-001:/input -v /home/ge.polymtl.ca/thdaga/docker_pl/output_sub001:/output  plbenveniste/ms-challenge-monomodal:2.0 -i /input -o /output
```

> [!NOTE]  
> the `--rm` flag makes sure that the docker is closed at the end  
> the `-v` flag mounts the volume, so that they are visible to the docker

Then push the Docker: 
```console
docker login -u plbenveniste
docker push plbenveniste/ms-challenge-monomodal:2.0
```

Then I created the boutique descriptor of my tool miccai2025_challenge_descriptor_neuropoly_monomodal.json
Then I also created the invocation file miccai2025_challenge_invocation_neuropoly_monomodal.json

Finally to validate everything, I did: 
- Install Boutiques: `pip install boutiques`
- Validate your descriptor: `bosh validate miccai2025_challenge_descriptor_neuropoly_monomodal.json`
- Execute the tool: `bosh exec launch miccai2025_challenge_descriptor_neuropoly_monomodal.json miccai2025_challenge_invocation_neuropoly_monomodal.json`