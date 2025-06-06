"""
This script removes lesions in predictions if the max value of the lesion mask is below a certain threshold.
"""
import json
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions in predictions if the max value of the lesion mask is below a certain threshold.")
    parser.add_argument('--thresh', type=float, required=True, help='Threshold for removing lesions.')
    return parser.parse_args()

def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_251_prep"

    args = parse_args()
    thresh = args.thresh

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    # iterate over the images
    for image in images:
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        print("Subject name", images[image]["subject_name"])

        # Build path to the lesion masks
        lesion_mask_t2 = os.path.join(sub_folder,"predictions", "t2w_segmentation_masked.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder,"predictions", "psir_segmentation_masked.nii.gz")

        # Build an output folder
        output_folder = os.path.join(sub_folder, f"output_rmv_lesion_{thresh}")
        os.makedirs(output_folder, exist_ok=True)

        # In each mask, for each lesion, we check the mask value of the lesion
        ## We first focus on the T2w lesion mask
        ### We enumerate the lesions in the T2w lesion mask by using connected components
        t2_mask_data = nib.load(lesion_mask_t2).get_fdata()
        instances_t2, nb_labels_t2 = ndimage.label(t2_mask_data)
        ### Now we want to split instances_t2 into individual masks for each lesion
        individual_instances_t2 = np.zeros((nb_labels_t2, *t2_mask_data.shape), dtype=np.float32)
        for i in range(1, nb_labels_t2+1):
            instance_i_t2 = np.zeros_like(t2_mask_data)
            instance_i_t2[instances_t2 == i] = 1
            individual_instances_t2[i-1] = instance_i_t2
        ### For each individual instance, we check the max value
        for i in range(1, nb_labels_t2+1):
            # We select the soft seg of the current lesion
            soft_seg_current_lesion = t2_mask_data * individual_instances_t2[i-1]
            # binarize the soft seg of the current lesion
            soft_seg_current_lesion_bin = (soft_seg_current_lesion > 0).astype(np.float32)
            # if the max value of the soft seg is below the threshold, we remove the lesion from the T2w mask
            if np.max(soft_seg_current_lesion) < thresh:
                t2_mask_data = t2_mask_data * (1 - soft_seg_current_lesion_bin)
        ### Save the modified T2w lesion mask
        modified_t2_mask_path = os.path.join(output_folder, f"t2w_segmentation_masked_rmvLesion{thresh}.nii.gz")
        nib.save(nib.Nifti1Image(t2_mask_data, nib.load(lesion_mask_t2).affine), modified_t2_mask_path)

        ## Now we focus on the PSIR lesion mask
        ### We enumerate the lesions in the PSIR lesion mask by using connected components
        psir_mask_data = nib.load(lesion_mask_psir).get_fdata()
        instances_psir, nb_labels_psir = ndimage.label(psir_mask_data)
        ### Now we want to split instances_psir into individual masks for each lesion
        individual_instances_psir = np.zeros((nb_labels_psir, *psir_mask_data.shape), dtype=np.float32)
        for i in range(1, nb_labels_psir+1):
            instance_i_psir = np.zeros_like(psir_mask_data)
            instance_i_psir[instances_psir == i] = 1
            individual_instances_psir[i-1] = instance_i_psir
        ### For each individual instance, we check the max value
        for i in range(1, nb_labels_psir+1):
            # We select the soft seg of the current lesion
            soft_seg_current_lesion = psir_mask_data * individual_instances_psir[i-1]
            # binarize the soft seg of the current lesion
            soft_seg_current_lesion_bin = (soft_seg_current_lesion > 0).astype(np.float32)
            # if the max value of the soft seg is below the threshold, we remove the lesion from the PSIR mask
            if np.max(soft_seg_current_lesion) < thresh:
                psir_mask_data = psir_mask_data * (1 - soft_seg_current_lesion_bin)
        ### Save the modified PSIR lesion mask
        modified_psir_mask_path = os.path.join(output_folder, f"psir_segmentation_masked_rmvLesion{thresh}.nii.gz")
        nib.save(nib.Nifti1Image(psir_mask_data, nib.load(lesion_mask_psir).affine), modified_psir_mask_path)

        break


if __name__ == "__main__":
    main()
