"""
This file is used to evaluate a semantic segmentation of the model.

Input:
    -pred: Predicted binary segmentation file
    -label: Label file with the ground truth
    -output: Path to the output folder where results will be saved
    -image-name: Name of the image to evaluate
    

Output:
    None

Example: 
    python evaluate_semantic_pred.py -pred /path/to/predictions -label /path/to/labels -output /path/to/output -image-name image_name

Author: Pierre-Louis Benveniste
"""

import os
import argparse
import json
import nibabel as nib
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity, normalised_surface_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", required=True, type=str, help="Predicted binary segmentation file")
    parser.add_argument("-label", required=True, type=str, help="Label file with the ground truth")
    parser.add_argument("-output", required=True, type=str, help="Path to the output folder where results will be saved")
    parser.add_argument("-image-name", required=True, type=str, help="Name of the image to evaluate")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    pred = args.pred
    label = args.label
    output_folder = args.output
    image_name = args.image_name

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the predictions and the label
    pred_data = nib.load(str(pred)).get_fdata()
    label_data = nib.load(str(label)).get_fdata()

    # Get resolution
    resolution = nib.load(str(pred)).header.get_zooms()

    # Compute dice score
    dice = dice_score(pred_data, label_data)
    ppv = lesion_ppv(label_data, pred_data)
    f1 = lesion_f1_score(label_data, pred_data)
    sensitivity = lesion_sensitivity(label_data, pred_data)
    nsd = normalised_surface_distance(pred_data, label_data, resolution)

    # Save all results in one array where columns are the metrics and row is the image name
    results = {
        "dice": dice,
        "ppv": ppv,
        "f1": f1,
        "sensitivity": sensitivity,
        "nsd": nsd
    }
    json_data = {}
    json_data[image_name] = results

    print(json_data)
    # Save the results in a json file
    json_file_path = os.path.join(output_folder,f"{image_name}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    return None


if __name__ == "__main__":
    main()