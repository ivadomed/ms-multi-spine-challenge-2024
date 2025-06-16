"""
This script removes lesions for which the max voxel value is below 0.8.

Input: 
    -subj_dict: the dictionnary of the subject
    -output_folder: the output folder where the segmentation masks will be saved

Output:
    -subj_dict: the updated dictionnary of the subject with the segmentation masks updated

Author: Thomas Dagonneau Pierre-Louis Benveniste
"""
import argparse
import os
import nibabel as nib
from scipy import ndimage
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions depending on lesion max voxel value")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


threshold = 0.4  # Threshold for lesion max voxel value



def remove_lesions_max_value(subj_dict, output_folder, threshold=0.3):
    """
    Remove lesions based on the soft mask's max value. Lesions with max value < threshold
    are removed from both the T2 binary segmentation and the soft segmentation.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the soft segmentation mask used to evaluate lesion intensity
    soft_mask_path = subj_dict['soft_mask_rmv_lesions_outside_sc']
    soft_mask_nii = nib.load(soft_mask_path)
    soft_mask_data = soft_mask_nii.get_fdata()

    # Load the binary segmentation mask (T2)
    t2_mask_path = subj_dict['t2_segmentation_file_rmv_lesions_outside_sc']
    t2_mask_nii = nib.load(t2_mask_path)
    t2_mask_data = t2_mask_nii.get_fdata()

    # Identify connected components in the soft segmentation
    instances, nb_labels = ndimage.label(soft_mask_data > 0)

    # Create masks for lesions to remove
    removal_mask = np.zeros_like(soft_mask_data, dtype=bool)

    for label_id in range(1, nb_labels + 1):
        lesion_mask = (instances == label_id)
        max_val = np.max(soft_mask_data[lesion_mask])
        if max_val < threshold:
            removal_mask |= lesion_mask  # Add this lesion to the removal mask

    # Remove the identified lesions from both masks
    new_soft_mask_data = soft_mask_data * (~removal_mask)
    new_t2_mask_data = t2_mask_data * (~removal_mask)

    # Save the cleaned masks
    soft_out_path = os.path.join(output_folder, "soft_segmentation_file_rmv_lesions_max_value.nii.gz")
    t2_out_path = os.path.join(output_folder, "t2_segmentation_file_rmv_lesions_max_value.nii.gz")
    nib.save(nib.Nifti1Image(new_soft_mask_data, soft_mask_nii.affine), soft_out_path)
    nib.save(nib.Nifti1Image(new_t2_mask_data, t2_mask_nii.affine), t2_out_path)

    # Update the dictionary
    subj_dict['soft_mask_rmv_lesions_max_value'] = soft_out_path
    subj_dict['t2_segmentation_file_rmv_lesions_max_value'] = t2_out_path

    return subj_dict



if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = remove_lesions_max_value(args.subj_dict, args.output_folder)