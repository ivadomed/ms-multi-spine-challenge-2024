# ms-spine-challenge
Deep learning models for the MS-Multi-Spine challenge 2024: https://portal.fli-iam.irisa.fr/ms-multi-spine/

More information can be found in the Zenodo record: https://zenodo.org/records/14051168


# Docker 

To build a docker image we first need a docker image to start with. 
Since nnUNet uses pythorch with `torch==2.5.1` I will use `docker pull pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel`
The `devel` stands for developpement and is bigger than the `runtime` but it's more complete and allows to run everything you want in pytorch. 
 