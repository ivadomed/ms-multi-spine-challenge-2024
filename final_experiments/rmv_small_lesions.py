"""
This script removes lesions in predictions depending on their sizes (in number of voxels).
A 0.5 threshold is applied for binarization.
"""
import json
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    pred_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep"

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    for min_volume in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 27, 30]:
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
            lesion_mask = os.path.join(sub_folder,"calibration", "merged_segmentation_masked_thresh_0.8.nii.gz")

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
                # If the lesion is smaller than the minimum volume, we remove it
                if np.sum(individual_instances[i-1]) < min_volume:
                    mask_data = mask_data * (1 - individual_instances[i-1])
            ### Save the modified T2w lesion mask
            modified_mask_path = os.path.join(output_folder, f"segmentation_masked_rmvLesion{min_volume}.nii.gz")
            nib.save(nib.Nifti1Image(mask_data, nib.load(lesion_mask).affine), modified_mask_path)


if __name__ == "__main__":
    main()
