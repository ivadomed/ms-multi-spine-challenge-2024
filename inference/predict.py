
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
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for inference (default=cuda).")
    args = parser.parse_args()
    return args

def run_inference(input, output, dataset_id, plans, trainer, configuration, checkpoint, fold, device):
    """
    Run the nnUNet inference on the dataset.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    #Create the output directory for the dataset-id if it doesn't exist
    dataset_output = os.path.join(output, f'Dataset{dataset_id}')
    os.makedirs(dataset_output, exist_ok=True)

    for fold in range(5):
        # Create the output directory for the fold if it doesn't exist
        fold_output = os.path.join(dataset_output, f'fold_{fold}')
        os.makedirs(fold_output, exist_ok=True)

        #Create the output directory specific to the model 
        dataset_output = os.path.join(dataset_output, f'{trainer}__{plans}__{configuration}_{fold}')
        os.makedirs(dataset_output, exist_ok=True)
        


        #Path to the input
        input_dir = os.path.join(input, f'Dataset{dataset_id}_MsMultiSpine/imagesTr')
        

        # Run inference 

        #assert os.system(f"nnUNetv2_predict -i {input_dir} -o {dataset_output}/prediction -d {dataset_id} -p {plans} -tr {trainer} -c {configuration} -f {fold} -chk {checkpoint} ") == 0

        assert os.system(f"nnUNetv2_predict -i {input_dir} -o {dataset_output}/prediction -d {dataset_id} -p {plans} -tr {trainer} -c {configuration} -f {fold} -chk {checkpoint} -device {device}") == 0


def voting(input_dir):
    # Base directory (where the folders are located)
    base_dir = input_dir

    # Prediction folders
    model_folders = [f"{i}" for i in range(5)]

    # Output folder
    output_folder = os.path.join(base_dir, "vote", "prediction")
    os.makedirs(output_folder, exist_ok=True)

    # Collect all subject filenames from the first model folder
    prediction_folder = os.path.join(base_dir, model_folders[0], "prediction")
    subject_files = sorted([f for f in os.listdir(prediction_folder) if f.endswith(".nii.gz")])

    print(f"Found {len(subject_files)} subjects to process...")

    # Voting function
    def majority_vote(volumes):
        stacked = np.stack(volumes, axis=0)  # Shape: (5, H, W, D)
        vote_sum = np.sum(stacked, axis=0)
        return (vote_sum >= 3).astype(np.uint8)

    # Process each subject
    for subject_file in subject_files:
        predictions = []
        affine, header = None, None

        for model_folder in model_folders:
            pred_path = os.path.join(base_dir, model_folder, "prediction", subject_file)
            pred_nib = nib.load(pred_path)
            pred_data = pred_nib.get_fdata().astype(np.uint8)

            if affine is None:
                affine = pred_nib.affine
                header = pred_nib.header

            predictions.append(pred_data)

        # Apply majority vote
        voted_mask = majority_vote(predictions)

        # Save result
        out_path = os.path.join(output_folder, subject_file)
        nib.save(nib.Nifti1Image(voted_mask, affine, header), out_path)


def main(): 

    # Parse arguments
    args = parse_arguments()

    # Run inference
    run_inference(args.input, args.output, args.dataset_id, args.plans, args.trainer, args.configuration, args.checkpoint, args.fold, args.device)

    voting(args.output)

    print(f"Inference completed. Results saved in {args.output}")


if __name__ == "__main__":
    main()





    