"""
This code aggregates the predictions of the nnUNet model for each fold and saves them in the same folder as the input data.
It does the averaging of the predictionsfor the binary and soft predictions.

Input: 
    --pred: Path to the prediction folder containing folders for each fold

Output:
    None

Example:
    python aggregate_predictions.py --pred /path/to/predictions

Author: Pierre-Louis Benveniste    
"""
import os
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate predictions from nnUNet')
    parser.add_argument('--pred', help='Path to the prediction folder', required=True)
    args = parser.parse_args()
    return args


def main():
    # Get arguments
    args = parse_args()
    pred_path = args.pred

    # Path to binary folders
    pred_folders_binary = []
    pred_folders_soft = []
    for i in range(5):
        pred_folders_binary.append(os.path.join(pred_path, f'fold_{i}_binary'))
        pred_folders_soft.append(os.path.join(pred_path, f'fold_{i}_soft'))

    # if the folders do not exist, we raise an error
    if not all(os.path.exists(f) for f in pred_folders_binary):
        raise ValueError(f"One or more folders do not exist: {pred_folders_binary}")
    
    # Build output folder
    output_soft_avg = os.path.join(pred_path, 'soft_avg')
    output_soft_avg_bin = os.path.join(pred_path, 'soft_avg_bin')
    os.makedirs(output_soft_avg, exist_ok=True)
    os.makedirs(output_soft_avg_bin, exist_ok=True)
    output_binary_avg = os.path.join(pred_path, 'binary_avg')
    output_binary_avg_bin = os.path.join(pred_path, 'binary_avg_bin')
    os.makedirs(output_binary_avg, exist_ok=True)
    os.makedirs(output_binary_avg_bin, exist_ok=True)

    # Get the list of files in the first fold using rglob
    pred_files = list(Path(pred_folders_binary[0]).rglob('*.nii.gz'))

    # Loop through each file
    for pred_file in tqdm(pred_files):
        # Get the file name and the corresponding image
        file_name = os.path.basename(pred_file)
        pred_soft = []
        pred_binary = []
        for i in range(5):
            pred_soft.append(os.path.join(pred_folders_soft[i], file_name))
            pred_binary.append(os.path.join(pred_folders_binary[i], file_name))

        # Load the predictions
        pred_soft = [nib.load(f).get_fdata() for f in pred_soft]
        pred_binary = [nib.load(f).get_fdata() for f in pred_binary]
        # Average the predictions
        pred_soft_avg = np.mean(pred_soft, axis=0)
        pred_binary_avg = np.mean(pred_binary, axis=0)
        # We apply thresholding
        pred_soft_avg[pred_soft_avg < 0.00001] = 0 # 0.0001 for soft predictions
        pred_soft_avg_bin = np.copy(pred_soft_avg)
        pred_soft_avg_bin[pred_soft_avg_bin > 0.5] = 1
        pred_soft_avg_bin[pred_soft_avg_bin <= 0.5] = 0
        pred_binary_avg_bin = np.copy(pred_binary_avg)
        pred_binary_avg_bin[pred_binary_avg_bin > 0.5] = 1
        pred_binary_avg_bin[pred_binary_avg_bin <= 0.5] = 0

        # Save the predictions
        pred_soft_avg_img = nib.Nifti1Image(pred_soft_avg, nib.load(os.path.join(pred_folders_soft[i], file_name)).affine)
        nib.save(pred_soft_avg_img, os.path.join(output_soft_avg, file_name))
        pred_soft_avg_bin_img = nib.Nifti1Image(pred_soft_avg_bin, nib.load(os.path.join(pred_folders_soft[i], file_name)).affine)
        nib.save(pred_soft_avg_bin_img, os.path.join(output_soft_avg_bin, file_name))
        pred_binary_avg_img = nib.Nifti1Image(pred_binary_avg, nib.load(os.path.join(pred_folders_binary[i], file_name)).affine)
        nib.save(pred_binary_avg_img, os.path.join(output_binary_avg, file_name))
        pred_binary_avg_bin_img = nib.Nifti1Image(pred_binary_avg_bin, nib.load(os.path.join(pred_folders_binary[i], file_name)).affine)
        nib.save(pred_binary_avg_bin_img, os.path.join(output_binary_avg_bin, file_name))
    print("All predictions aggregated successfully!")


if __name__ == '__main__':
    main()