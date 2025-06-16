# Creation of the virtual environment

```console
python3 -m venv .venv_nnunet
source .venv_nnunet/bin/activate
```

# Installing libraries required

```console
pip3 install torch torchvision torchaudio
```

# In my fork of nnUNet
```console
pip install -e .
pip install triton
```

# Other usefull commands

```console
export nnUNet_raw="/home/p/plb/links/projects/aip-jcohen/plb/challenge/nnUNet_raw"
export nnUNet_preprocessed="/home/p/plb/links/scratch/challenge/nnUNet_preprocessed"
export nnUNet_results="/home/p/plb/links/scratch/challenge/nnUNet_results"
```