"""
This script evaluates the performance of removal of lesions based on their max voxel value.

Author: Pierre-Louis Benveniste
"""
import json
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity, normalised_surface_distance
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the performance of removal of lesions based on their max voxel value.")
    parser.add_argument("--image_dict", type=str, required=True, help="Path to the JSON file containing image metadata.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing subject folders with predictions.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where results will be saved.")
    return parser.parse_args()


def main():

    args = parse_args()
    image_dict = args.image_dict
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    for thresh in [0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        print(f"Evaluating threshold {thresh}")
        # Dict of dice score
        dice_scores = {}
        ppv_scores = {}
        f1_scores = {}
        sensitivity_scores = {}
        nsd_scores = {}

        # iterate over the images
        for image in tqdm(images):
            # If contrast is T2, we skip
            if images[image]["contrast"] == "T2w":
                continue
            # Build a folder for each subject
            sub_folder = os.path.join(input_folder, images[image]["subject_name"])
            # print("Subject name", images[image]["subject_name"])

            # Build path to the lesion masks
            lesion_mask_t2 = os.path.join(sub_folder,f"output_rmv_lesion_{thresh}", f"t2w_segmentation_masked_rmvLesion{thresh}.nii.gz")
            lesion_mask_psir = os.path.join(sub_folder,f"output_rmv_lesion_{thresh}", f"psir_segmentation_masked_rmvLesion{thresh}.nii.gz")

            # Build path to the ground truth
            ground_truth = images[image]["t2w_raw_label_file"]

            # Load the predictions and the label
            pred_data_t2 = nib.load(str(lesion_mask_t2)).get_fdata()
            pred_data_psir = nib.load(str(lesion_mask_psir)).get_fdata()
            label_data = nib.load(str(ground_truth)).get_fdata()

            # Binarize the predictions
            pred_data_t2 = (pred_data_t2 > 0).astype(np.float32)
            pred_data_psir = (pred_data_psir > 0).astype(np.float32)
            # Binarize the label data as well
            label_data = (label_data > 0).astype(np.float32)

            # Get resolution
            resolution = nib.load(str(images[image]["t2w_raw_image"])).header.get_zooms()

            # Compute dice score
            dice_t2 = dice_score(pred_data_t2, label_data)
            dice_psir = dice_score(pred_data_psir, label_data)
            ppv_t2 = lesion_ppv(label_data, pred_data_t2)
            ppv_psir = lesion_ppv(label_data, pred_data_psir)
            f1_t2 = lesion_f1_score(label_data, pred_data_t2)
            f1_psir = lesion_f1_score(label_data, pred_data_psir)
            sensitivity_t2 = lesion_sensitivity(label_data, pred_data_t2)
            sensitivity_psir = lesion_sensitivity(label_data, pred_data_psir)
            nsd_t2 = normalised_surface_distance(label_data, pred_data_t2, resolution)
            nsd_psir = normalised_surface_distance(label_data, pred_data_psir, resolution)

            image_name_psir = image 
            image_name_t2 = image.replace(f"_{images[image]['contrast']}", "_T2w")

            dice_scores[image_name_t2] = dice_t2
            ppv_scores[image_name_t2] = ppv_t2
            f1_scores[image_name_t2] = f1_t2
            sensitivity_scores[image_name_t2] = sensitivity_t2
            nsd_scores[image_name_t2] = nsd_t2

            dice_scores[image_name_psir] = dice_psir
            ppv_scores[image_name_psir] = ppv_psir
            f1_scores[image_name_psir] = f1_psir
            sensitivity_scores[image_name_psir] = sensitivity_psir
            nsd_scores[image_name_psir] = nsd_psir

        # Save the results
        os.makedirs(os.path.join(output_folder, f"rmv_{thresh}"), exist_ok=True)
        with open(os.path.join(output_folder, f"rmv_{thresh}", f"dice_scores.txt"), "w") as f:
            for key, value in dice_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"rmv_{thresh}", f"ppv_scores.txt"), "w") as f:
            for key, value in ppv_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"rmv_{thresh}", f"f1_scores.txt"), "w") as f:
            for key, value in f1_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"rmv_{thresh}", f"sensitivity_scores.txt"), "w") as f:
            for key, value in sensitivity_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"rmv_{thresh}", f"nsd_scores.txt"), "w") as f:
            for key, value in nsd_scores.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()