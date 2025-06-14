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


min_volume = 50  # Threshold in voxel unit to remove small lesions


def remove_small_lesions(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Build the output path
    os.makedirs(output_folder, exist_ok=True)

    # Build input mask path
    lesion_mask = subj_dict['binarized_lesion_mask']

    ## In the lesion mask we enumerate the lesions by using connected components
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
    lesion_mask_rmv_small_lesion = os.path.join(output_folder, "lesion_mask_rmv_small_lesion.nii.gz")
    nib.save(nib.Nifti1Image(mask_data, nib.load(lesion_mask).affine), lesion_mask_rmv_small_lesion)
    
    # Update the subject dictionary with the new path
    subj_dict['lesion_mask_rmv_small_lesion'] = lesion_mask_rmv_small_lesion

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = remove_small_lesions(args.subj_dict, args.output_folder)