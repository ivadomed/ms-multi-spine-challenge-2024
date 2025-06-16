"""
This scripts computes the lesion_probbaility for each lesion instance.

Author: Pierre-Louis Benveniste
"""
import os
import nibabel as nib
from scipy import ndimage
import numpy as np
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Compute lesion probabilities for each lesion instance.")
    parser.add_argument('--image_dict', type=str, required=True, help='Path to the JSON file containing image metadata.')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing subject folders with predictions.')
    return parser.parse_args()


def main():

    args = parse_args()
    image_dict = args.image_dict
    pred_folder = args.pred_folder

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    lesion_probabilities = []

    # iterate over the images
    for image in tqdm(images):
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(pred_folder, images[image]["subject_name"])
        # print("Subject name", images[image]["subject_name"])
        # Load the lesion mask
        lesion_mask_bin = os.path.join(sub_folder, "output_rmv_small_lesion", "segmentation_masked_rmvLesion18.nii.gz")
        lesion_mask_soft = os.path.join(sub_folder, "lesion_mask_rmv_lesion0p8_merged.nii.gz")

        # Multiply the bin mask by the soft mask to get the final soft segmentation mask
        output_mask = os.path.join(sub_folder, "final_mask_soft.nii.gz")
        assert os.system(f"sct_maths -i {lesion_mask_bin} -mul {lesion_mask_soft} -o {output_mask}") == 0

        # Compute the lesion probability for each instance
        mask_data = nib.load(output_mask).get_fdata()
        instances, nb_labels = ndimage.label(mask_data)
        print(f"Number of labels: {nb_labels}")
        ## For each instance, we want to extract the soft segmentation mask of the lesion
        individual_instances = np.zeros((nb_labels, *mask_data.shape), dtype=np.float32)
        for i in range(1, nb_labels + 1):
            instance_i = np.zeros_like(mask_data)
            instance_i[instances == i] = 1
            individual_instances[i - 1] = instance_i * mask_data
        ## for each individual instance, we compute the lesion probability
        for i in range(1, nb_labels + 1):
            # Compute the lesion probability as the mean value of the soft segmentation mask
            lesion_probability = np.mean(individual_instances[i - 1][individual_instances[i - 1] > 0])
            lesion_probabilities.append({
                "subject_name": images[image]["subject_name"],
                "lesion_id": i,
                "lesion_probability": float(lesion_probability)
            })

    # Save the lesion probabilities to a json file
    output_json = os.path.join(pred_folder, "lesion_probabilities.json")
    with open(output_json, "w") as f:
        json.dump(lesion_probabilities, f)
    
    # Now we want to extract only the lesion probabilities for each subject
    list_lesion_probabilities = [lesion["lesion_probability"] for lesion in lesion_probabilities]
    
    # Now we want to plot the histogram of the lesion probabilities
    plt.hist(list_lesion_probabilities, bins=50, density=True)
    plt.xlabel("Lesion Probability")
    plt.ylabel("Density")
    plt.title("Histogram of Lesion Probabilities")
    plt.savefig(os.path.join(pred_folder, "lesion_probabilities_histogram.png"))

    
if __name__ == "__main__":
    main()