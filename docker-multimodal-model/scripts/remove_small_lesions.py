"""
This script removes the lesions for which the volume is below a certain threshold (in voxel units)

Input: 
    -subj_dict: the dictionnary of the subject
    -output_folder: the output folder where the segmentation masks will be saved

Output:
    -subj_dict: the updated dictionnary of the subject with the segmentation masks updated

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import nibabel as nib
import numpy as np
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(description="Remove small lesions script")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


min_volume = 18  # Threshold in voxel unit to remove small lesions


def remove_small_lesions(subj_dict, output_folder, min_volume=50):
    """
    Removes lesions smaller than `min_volume` voxels from both the binary T2 segmentation
    and the soft segmentation (after lesion max filtering).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load masks
    t2_path = subj_dict['t2_segmentation_file_rmv_lesions_max_value']
    soft_path = subj_dict['soft_mask_rmv_lesions_max_value']

    t2_nii = nib.load(t2_path)
    t2_data = t2_nii.get_fdata()

    soft_nii = nib.load(soft_path)
    soft_data = soft_nii.get_fdata()

    # Identify connected components in the T2 segmentation (binary)
    instances, nb_labels = ndimage.label(t2_data > 0)

    # Build a removal mask
    removal_mask = np.zeros_like(t2_data, dtype=bool)

    for label in range(1, nb_labels + 1):
        lesion = (instances == label)
        if np.sum(lesion) < min_volume:
            removal_mask |= lesion

    # Remove small lesions from both masks
    t2_cleaned = t2_data * (~removal_mask)
    soft_cleaned = soft_data * (~removal_mask)

    # Save cleaned masks
    t2_out = os.path.join(output_folder, "t2_segmentation_file_rmv_small_lesions.nii.gz")
    soft_out = os.path.join(output_folder, "soft_segmentation_file_rmv_small_lesions.nii.gz")

    nib.save(nib.Nifti1Image(t2_cleaned, t2_nii.affine), t2_out)
    nib.save(nib.Nifti1Image(soft_cleaned, soft_nii.affine), soft_out)

    # Update dictionary
    subj_dict['t2_segmentation_file_rmv_small_lesions'] = t2_out
    subj_dict['soft_mask_rmv_small_lesions'] = soft_out

    return subj_dict



if __name__ == "__main__":
    args = parse_args()
   