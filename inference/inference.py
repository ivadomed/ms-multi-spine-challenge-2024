"""
This script is used to convert the ms-multi-spine challenge dataset from the BIDS format to the nnUNet format.
The script will need to create the types of dataset detailed in the PR head: https://github.com/ivadomed/ms-multi-spine-challenge-2024/pull/13.
It takes as input a predefined data split (detailed in this issue: https://github.com/ivadomed/ms-multi-spine-challenge-2024/issues/12). 

Input: 
    --data: Path to the root directory of the dataset.
    --output: Path to the output directory where the nnUNet dataset will be created (nnUnet_raw).
    --path-data-split: Path to the data split YAML file.
    --task-name: Name of the task for nnUNet.
    --task-number: Number of the task for nnUnet (it has to be 3 digits).
    --dataset-type: Number corresponding to the dataset type we want to create. 

Output:
    None

Author: Thomas Dagonneau and Pierre-Louis Benveniste

TODO: 
    - Mofidy the part above TODO. 
    - Add the compute_metrics function to compute the metrics for the nnUNet dataset.
"""
import os
import json
import nibabel as nib
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input directory containing the dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory where the inference results will be stored.")
    parser.add_argument("--dataset-id", type=str, required=True, help="nnUNet dataset id.")
    parser.add_argument("--plans", type=str, default="nnUNetResEncUNetLPlans", help="Plans identifier for nnUNet.")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer", help="Trainer class for nnUNet.")
    parser.add_argument("--configuration", type=str, required=True, help="Configuration for nnUNet.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="Checkpoint name for nnUNet.")
    parser.add_argument("--fold", type=str, default=all, help="Fold for nnUNet (default=all).")
    parser.add_argument("--train", type=bool, default=False, help="Specify if you want to make inference on train or test.")
    args = parser.parse_args()
    return args

def run_inference(input, output, dataset_id, plans, trainer, configuration, checkpoint, fold, train):
    """
    Run the nnUNet inference on the dataset.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    #Create the output directory for the dataset-id if it doesn't exist
    dataset_output = os.path.join(output, f'Dataset{dataset_id}')
    os.makedirs(dataset_output, exist_ok=True)

    #Create the output directory specific to the model 
    dataset_output = os.path.join(dataset_output, f'{trainer}__{plans}__{configuration}_{fold}')
    os.makedirs(dataset_output, exist_ok=True)
    

    if train: 

        #Path to the input
        input_dir = os.path.join(input, f'Dataset{dataset_id}_MsMultiSpine/imagesTr')
        
        #Path to the labels 
        label_dir = os.path.join(input, f'Dataset{dataset_id}_MsMultiSpine','labelsTr')

    else: 

        #Path to the input
        input_dir = os.path.join(input, f'Dataset{dataset_id}_MsMultiSpine/imagesTs')
        
        #Path to the labels 
        label_dir = os.path.join(input, f'Dataset{dataset_id}_MsMultiSpine','labelsTs')

    # Run inference 

    #assert os.system(f"nnUNetv2_predict -i {input_dir} -o {dataset_output}/prediction -d {dataset_id} -p {plans} -tr {trainer} -c {configuration} -f {fold} -chk {checkpoint} ") == 0

    assert os.system(f"nnUNetv2_predict -i {input_dir} -o {dataset_output}/prediction -d {dataset_id} -p {plans} -tr {trainer} -c {configuration} -f {fold} -chk {checkpoint} --save_probabilities") == 0
 
    
            



def main(): 

    # Parse arguments
    args = parse_arguments()

    # Run inference
    run_inference(args.input, args.output, args.dataset_id, args.plans, args.trainer, args.configuration, args.checkpoint, args.fold, args.train)

    print(f"Inference completed. Results saved in {args.output}")


if __name__ == "__main__":
    main()