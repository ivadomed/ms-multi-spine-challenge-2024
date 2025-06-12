"""
This script performs calibration of the threshold during mask binarization.
"""
import json
import os
from tqdm import tqdm

def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep"

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    # iterate over the images
    for image in tqdm(images):
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        print("Subject name", images[image]["subject_name"])

        # Build path to the lesion masks
        mask = os.path.join(sub_folder,"lesion_mask_rmv_lesion0p8_merged.nii.gz")

        # Build an output folder
        output_folder_calibration = os.path.join(sub_folder, "calibration_after_rmv_lesion0p8")
        os.makedirs(output_folder_calibration, exist_ok=True)

        for thresh in [0.001, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            # We build the output path
            output_path = os.path.join(output_folder_calibration, f"merged_segmentation_masked_thresh_{thresh}.nii.gz")
            # We perform the binarization
            assert os.system(f"sct_maths -i {mask} -bin {thresh} -o {output_path} ") == 0


if __name__ == "__main__":
    main()