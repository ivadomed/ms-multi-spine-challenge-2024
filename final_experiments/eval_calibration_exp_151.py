"""
This script evaluates the performance of the calibration of the model
"""
import json
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity, normalised_surface_distance
from tqdm import tqdm


def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    input_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep"

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep_calibration"

    # bUild the output folder
    os.makedirs(output_folder, exist_ok=True)

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    for thresh in [0.001, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        # Dict of dice score
        dice_scores = {}
        ppv_scores = {}
        f1_scores = {}
        sensitivity_scores = {}
        nsd_scores = {}
        # Print status
        print (f"Processing threshold: {thresh}")

        # iterate over the images
        for image in tqdm(images):
            # If contrast is T2, we skip
            if images[image]["contrast"] == "T2w":
                continue
            # Build a folder for each subject
            sub_folder = os.path.join(input_folder, images[image]["subject_name"])
            # print("Subject name", images[image]["subject_name"])

            # Build the path to the lesion mask
            lesion_mask = os.path.join(sub_folder, "calibration_after_rmv_lesion0p8", f"merged_segmentation_masked_thresh_{thresh}.nii.gz")

            # Build path to the ground truth
            ground_truth = images[image]["t2w_raw_label_file"]

            # Load the predictions and the label
            pred_data = nib.load(str(lesion_mask)).get_fdata()
            label_data = nib.load(str(ground_truth)).get_fdata()

            # Get resolution
            resolution = nib.load(str(images[image]["t2w_raw_image"])).header.get_zooms()

            # Compute dice score
            dice = dice_score(pred_data, label_data)
            ppv = lesion_ppv(label_data, pred_data)
            f1 = lesion_f1_score(label_data, pred_data)
            sensitivity = lesion_sensitivity(label_data, pred_data)
            nsd = normalised_surface_distance(pred_data, label_data, resolution)

            image_name = image.replace(f"_{images[image]['contrast']}", "_T2w")

            # Save the dice score
            dice_scores[image_name] = dice
            ppv_scores[image_name] = ppv
            f1_scores[image_name] = f1
            sensitivity_scores[image_name] = sensitivity
            nsd_scores[image_name] = nsd

        # Save the results
        os.makedirs(os.path.join(output_folder, f"calib_{thresh}"), exist_ok=True)
        with open(os.path.join(output_folder, f"calib_{thresh}", f"dice_scores.txt"), "w") as f:
            for key, value in dice_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"calib_{thresh}", f"ppv_scores.txt"), "w") as f:
            for key, value in ppv_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"calib_{thresh}", f"f1_scores.txt"), "w") as f:
            for key, value in f1_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"calib_{thresh}", f"sensitivity_scores.txt"), "w") as f:
            for key, value in sensitivity_scores.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"calib_{thresh}", f"nsd_scores.txt"), "w") as f:
            for key, value in nsd_scores.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()