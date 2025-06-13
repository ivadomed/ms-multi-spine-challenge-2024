"""
This script evaluates the performance increase of the post-processing step which removes the lesions outside the spinal cord.
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

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep_rmv_lesion_outside_spinal_cord"

    # bUild the output folder
    os.makedirs(output_folder, exist_ok=True)

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}
    
    # Dict of scores
    dice_scores = {}
    ppv_scores = {}
    f1_scores = {}
    sensitivity_scores = {}
    nsd_scores = {}

    # Dict of scores post-processed
    dice_scores_postproc = {}
    ppv_scores_postproc = {}
    f1_scores_postproc = {}
    sensitivity_scores_postproc = {}
    nsd_scores_postproc = {}

    # iterate over the images
    for image in tqdm(images):
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(input_folder, images[image]["subject_name"])
        # print("Subject name", images[image]["subject_name"])

        # Build the path to the lesion mask
        lesion_mask_t2w = os.path.join(sub_folder, "predictions", "t2w_segmentation.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder, "predictions", "psir_segmentation.nii.gz")
        lesion_mask_t2w_postproc = os.path.join(sub_folder, "predictions", "t2w_segmentation_masked.nii.gz")
        lesion_mask_psir_postproc = os.path.join(sub_folder, "predictions", "psir_segmentation_masked.nii.gz")

        # Build path to the ground truth
        ground_truth = images[image]["t2w_raw_label_file"]

        # Load the predictions and the label
        pred_data_t2w = nib.load(lesion_mask_t2w).get_fdata()
        pred_data_psir = nib.load(lesion_mask_psir).get_fdata()
        label_data = nib.load(ground_truth).get_fdata()
        pred_data_t2w_postproc = nib.load(lesion_mask_t2w_postproc).get_fdata()
        pred_data_psir_postproc = nib.load(lesion_mask_psir_postproc).get_fdata()

        # We should first binarize the masks
        pred_data_t2w = (pred_data_t2w > 0).astype(np.float32)
        pred_data_psir = (pred_data_psir > 0).astype(np.float32)
        pred_data_t2w_postproc = (pred_data_t2w_postproc > 0).astype(np.float32)
        pred_data_psir_postproc = (pred_data_psir_postproc > 0).astype(np.float32)
        # We need to binarize the mask because it is a instance seg
        label_data = (label_data > 0).astype(np.float32)

        # Get resolution
        resolution = nib.load(str(images[image]["t2w_raw_image"])).header.get_zooms()

        # Compute dice score t2w
        dice = dice_score(pred_data_t2w, label_data)
        ppv = lesion_ppv(label_data, pred_data_t2w)
        f1 = lesion_f1_score(label_data, pred_data_t2w)
        sensitivity = lesion_sensitivity(label_data, pred_data_t2w)
        nsd = normalised_surface_distance(pred_data_t2w, label_data, resolution)
        # Get image name
        image_name = image.replace(f"_{images[image]['contrast']}", "_T2w")
        # Save the dice score
        dice_scores[image_name] = dice
        ppv_scores[image_name] = ppv
        f1_scores[image_name] = f1
        sensitivity_scores[image_name] = sensitivity
        nsd_scores[image_name] = nsd
        
        # For the psir image
        dice = dice_score(pred_data_psir, label_data)
        ppv = lesion_ppv(label_data, pred_data_psir)
        f1 = lesion_f1_score(label_data, pred_data_psir)
        sensitivity = lesion_sensitivity(label_data, pred_data_psir)
        nsd = normalised_surface_distance(pred_data_psir, label_data, resolution)
        # Get image name
        image_name = image
        # Save the dice score
        dice_scores[image_name] = dice
        ppv_scores[image_name] = ppv
        f1_scores[image_name] = f1
        sensitivity_scores[image_name] = sensitivity
        nsd_scores[image_name] = nsd

        # Now compute the post-processed scores
        # Compute dice score t2w post-processed
        dice_postproc = dice_score(pred_data_t2w_postproc, label_data)
        ppv_postproc = lesion_ppv(label_data, pred_data_t2w_postproc)
        f1_postproc = lesion_f1_score(label_data, pred_data_t2w_postproc)
        sensitivity_postproc = lesion_sensitivity(label_data, pred_data_t2w_postproc)
        nsd_postproc = normalised_surface_distance(pred_data_t2w_postproc, label_data, resolution)
        # Get image name
        image_name = image.replace(f"_{images[image]['contrast']}", "_T2w")
        # Save the dice score
        dice_scores_postproc[image_name] = dice_postproc
        ppv_scores_postproc[image_name] = ppv_postproc
        f1_scores_postproc[image_name] = f1_postproc
        sensitivity_scores_postproc[image_name] = sensitivity_postproc
        nsd_scores_postproc[image_name] = nsd_postproc
        # For the psir image post-processed
        dice_postproc = dice_score(pred_data_psir_postproc, label_data)
        ppv_postproc = lesion_ppv(label_data, pred_data_psir_postproc)
        f1_postproc = lesion_f1_score(label_data, pred_data_psir_postproc)
        sensitivity_postproc = lesion_sensitivity(label_data, pred_data_psir_postproc)
        nsd_postproc = normalised_surface_distance(pred_data_psir_postproc, label_data, resolution)
        # Get image name
        image_name = image
        # Save the dice score
        dice_scores_postproc[image_name] = dice_postproc
        ppv_scores_postproc[image_name] = ppv_postproc
        f1_scores_postproc[image_name] = f1_postproc
        sensitivity_scores_postproc[image_name] = sensitivity_postproc
        nsd_scores_postproc[image_name] = nsd_postproc

    # Save the results
    os.makedirs(os.path.join(output_folder, "before_removal"), exist_ok=True)
    with open(os.path.join(output_folder, "before_removal", f"dice_scores.txt"), "w") as f:
        for key, value in dice_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "before_removal", f"ppv_scores.txt"), "w") as f:
        for key, value in ppv_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "before_removal", f"f1_scores.txt"), "w") as f:
        for key, value in f1_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "before_removal", f"sensitivity_scores.txt"), "w") as f:
        for key, value in sensitivity_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "before_removal", f"nsd_scores.txt"), "w") as f:
        for key, value in nsd_scores.items():
            f.write(f"{key}: {value}\n")
    # Save the post-processed results
    os.makedirs(os.path.join(output_folder, "after_removal"), exist_ok=True)
    with open(os.path.join(output_folder, "after_removal", f"dice_scores.txt"), "w") as f:
        for key, value in dice_scores_postproc.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "after_removal", f"ppv_scores.txt"), "w") as f:
        for key, value in ppv_scores_postproc.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "after_removal", f"f1_scores.txt"), "w") as f:
        for key, value in f1_scores_postproc.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "after_removal", f"sensitivity_scores.txt"), "w") as f:
        for key, value in sensitivity_scores_postproc.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "after_removal", f"nsd_scores.txt"), "w") as f:
        for key, value in nsd_scores_postproc.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()