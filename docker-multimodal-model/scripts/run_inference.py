"""
This script runs the inference on the images using the 5 folds of the model.

Input: 
    -input_image: path to the input image
    -model_path: path to the model folder
    -output_folder: path to the output folder

Returns: 
    -output_image: path to the output image

Author: Pierre-Louis Benveniste     
"""
import argparse
import os
from pathlib import Path
import torch
import nibabel as nib
import numpy as np



# Import for nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input folder")
    parser.add_argument("-t", "--type", type=str, required=True, help="Type of contrast among PSIR STIR MP2RAGE")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()



def majority_vote(volumes, t):
    stacked = np.stack(volumes, axis=0)  # Shape: (5, H, W, D)
    vote_sum = np.sum(stacked, axis=0)
    if t == "STIR":
        return (vote_sum >= 1).astype(np.uint8)
    else : 
        return (vote_sum >= 3).astype(np.uint8)


def run_inference(input_image, output_folder, t):

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_inference")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Step 1: Run inference for each fold

    for fold in range(5):
        # Run inference for the fold

        assert os.system(f"nnUNetv2_predict -i {input_image} -o {output_folder}/fold_{fold}/prediction -d 200 -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDA5_150epochs -c 2d -f {fold} -chk checkpoint_best_{fold}.pth -device cpu") == 0
    

    # Step 2: Collect subject files from all folds
    # Assumes all folds predict the same files; we'll collect from fold_0
    subject_files = sorted([
        f for f in os.listdir(os.path.join(output_folder, "fold_0", "prediction"))
        if f.endswith(".nii.gz")
    ])

    print(f"Found {len(subject_files)} subjects to process for voting...")

    voted_output_folder = output_folder

    for subject_file in subject_files:
        predictions = []
        affine, header = None, None

        for fold in range(5):
            fold_pred_path = os.path.join(output_folder, f"fold_{fold}", "prediction", subject_file)

            if not os.path.exists(fold_pred_path):
                raise FileNotFoundError(f"Missing prediction for {subject_file} in fold {fold}")

            pred_nib = nib.load(fold_pred_path)
            pred_data = pred_nib.get_fdata().astype(np.uint8)

            if affine is None:
                affine = pred_nib.affine
                header = pred_nib.header

            predictions.append(pred_data)

        # ✅ Apply correct majority vote over 5 folds
        voted_mask = majority_vote(predictions, t)

        out_path = os.path.join(voted_output_folder, subject_file)
        nib.save(nib.Nifti1Image(voted_mask, affine, header), out_path)

        print(f"Saved voted mask for {subject_file}")

    return out_path



if __name__ == "__main__":
    args = parse_args()
    output_image = run_inference(args.input_image, args.output_folder)