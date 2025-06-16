"""
This script removes lesions in predictions depending on their sizes (in number of voxels).
A 0.5 threshold is applied for binarization.

Author: Pierre-Louis Benveniste
"""
import json
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Remove small lesions in predictions based on their size.")
    parser.add_argument('--image_dict', type=str, required=True, help='Path to the JSON file containing image metadata.')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing predictions.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder where results will be saved.')
    return parser.parse_args()


def main():

    args = parse_args()
    # Get the arguments
    image_dict = args.image_dict
    pred_folder = args.pred_folder
    final_output_folder = args.output_folder

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    lesion_sizes=[]

    for min_volume in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        print(f"Processing minimum volume: {min_volume} voxels")

        # iterate over the images
        for image in tqdm(images):
            # If contrast is T2, we skip
            if images[image]["contrast"] == "T2w":
                continue
            # Build a folder for each subject
            sub_folder = os.path.join(pred_folder, images[image]["subject_name"])
            # print("Subject name", images[image]["subject_name"])

            # Build path to the merged lesion mask
            lesion_mask = os.path.join(sub_folder,"calibration_after_rmv_lesion0p8", "merged_segmentation_masked_thresh_0.5.nii.gz")

            # Build an output folder
            output_folder = os.path.join(sub_folder, f"output_rmv_small_lesion")
            os.makedirs(output_folder, exist_ok=True)

            # In the lesion mask we enumerate the lesions by using connected components
            mask_data = nib.load(lesion_mask).get_fdata()
            instances, nb_labels = ndimage.label(mask_data)
            ### Now we want to split instances into individual masks for each lesion
            individual_instances = np.zeros((nb_labels, *mask_data.shape), dtype=np.float32)
            for i in range(1, nb_labels+1):
                instance_i = np.zeros_like(mask_data)
                instance_i[instances == i] = 1
                individual_instances[i-1] = instance_i
            ### For each individual instance, we check the number of voxels in the lesion
            for i in range(1, nb_labels+1):
                if min_volume ==0:
                    lesion_sizes.append(np.sum(individual_instances[i-1]))
                # If the lesion is smaller than the minimum volume, we remove it
                if np.sum(individual_instances[i-1]) < min_volume:
                    mask_data = mask_data * (1 - individual_instances[i-1])
            ### Save the modified T2w lesion mask
            modified_mask_path = os.path.join(output_folder, f"segmentation_masked_rmvLesion{min_volume}.nii.gz")
            nib.save(nib.Nifti1Image(mask_data, nib.load(lesion_mask).affine), modified_mask_path)
    
    os.makedirs(final_output_folder, exist_ok=True)
    # Save the csv file file with the lesion sizes
    lesion_sizes_path = os.path.join(final_output_folder, "lesion_sizes.csv")
    with open(lesion_sizes_path, 'w') as f:
        f.write("Lesion Size (voxels)\n")
        for size in lesion_sizes:
            f.write(f"{size}\n")
    # print mean, max, min, std and median in a txt file which we save
    txt_file = os.path.join(final_output_folder, 'lesion_sizes.txt')
    with open(txt_file, 'w') as f:
        f.write(f"Mean: {np.mean(lesion_sizes)}\n")
        f.write(f"Max: {np.max(lesion_sizes)}\n")
        f.write(f"Min: {np.min(lesion_sizes)}\n")
        f.write(f"Std: {np.std(lesion_sizes)}\n")
        f.write(f"Median: {np.median(lesion_sizes)}\n")
    # Plot the histogram of the lesion sizes
    plt.hist(lesion_sizes, bins=50, density=True)
    plt.xlabel("Lesion Size (voxels)")
    plt.ylabel("Density")
    plt.title("Histogram of Lesion Sizes")
    plt.savefig(os.path.join(final_output_folder, "lesion_sizes_histogram.png"))

if __name__ == "__main__":
    main()
