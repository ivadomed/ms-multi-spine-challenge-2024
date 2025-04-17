"""
This script evaluates instance segmentations performed by a model. 

Input:
    -pred: A segmentation with instance segmentations (each lesion is a separate instance)
    -pred_csv: A CSV file containing the prediction results (the probability of each instance)
    -gt: A segmentation with instance segmentations (each lesion is a separate instance)
    -output: Path to the output folder where results will be saved

Output:
    -results: A dictionary containing the evaluation results

Example usage: 
    python evaluate_instance_segmentation.py --pred pred.nii.gz --pred_csv pred.csv --gt gt.nii.gz --output ./output

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import shutil
from pathlib import Path
import nibabel as nib
from scipy import ndimage
from challenge_functions.lesion_level import generate_lesion_level_files
from challenge_functions.froc_calculation import calculate_froc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate instance segmentation.")
    parser.add_argument("--pred", type=str, required=True, help="Path to the predicted segmentation file.")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to the predicted CSV file.")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth segmentation file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder.")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    pred_path = args.pred
    pred_csv_path = args.pred_csv
    gt_path = args.gt
    output_path = args.output

    # Create the output folder
    os.makedirs(output_path, exist_ok=True)
    # Create a temp folder in the output folder
    temp_path = os.path.join(output_path, "temp")
    os.makedirs(temp_path, exist_ok=True)
    pred_temp_folder = os.path.join(temp_path, "pred")
    os.makedirs(pred_temp_folder, exist_ok=True)
    gt_temp_folder = os.path.join(temp_path, "gt")
    os.makedirs(gt_temp_folder, exist_ok=True)

    # For each .nii.gz file in the pred folder, copy it to the temp folder 
    list_pred = list(Path(pred_path).rglob("*.nii.gz"))
    for pred_file in list_pred:
        # Copy the file to the temp folder
        pred_file = str(pred_file)
        pred_file_name = os.path.basename(pred_file)
        temp_pred_file = os.path.join(pred_temp_folder, pred_file_name)
        shutil.copy(pred_file, temp_pred_file)

    # For the GT folder, for each .nii.gz file, we label the lesions with different instances and save them in the temp folder with the .nii extension
    list_gt = list(Path(gt_path).rglob("*.nii.gz"))
    for gt_file in list_gt:
        gt_file = str(gt_file)
        gt_file_name = os.path.basename(gt_file)
        gt_temp_file = os.path.join(gt_temp_folder, gt_file_name)
        instance_gt = nib.load(gt_file).get_fdata()
        instance_gt, nb_labels = ndimage.label(instance_gt)
        instance_gt = instance_gt.astype("uint16")
        # Save 
        nib.save(nib.Nifti1Image(instance_gt, nib.load(gt_file).affine), gt_temp_file)

    ## Here is what we want to replicate
    # generate_lesion_level_files(args, args.gt_path, args.pred_path, args.csv_path, args.iou_tresh, args.output_path)
    # calculate_froc(args, args.gt_tsv_path, args.pred_tsv_path, args.target_metric_list, args.metric_type, args.output_path, args.iou_tresh)
    # Let's do it
    generate_lesion_level_files(args, gt_temp_folder, pred_temp_folder, pred_csv_path, 0.2, temp_path)
    args.debug = False
    calculate_froc(args, os.path.join(temp_path, "table_gt.tsv"), os.path.join(temp_path, "table_pred.tsv"), [0.25, 0.5, 1, 2, 3], "mean_fp_per_image", temp_path, 0.2)


if __name__ == "__main__":
    main()