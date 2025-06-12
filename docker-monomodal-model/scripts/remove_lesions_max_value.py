"""
This script removes lesions for which the max voxel value is below 0.8.

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
from scipy import ndimage
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions depending on lesion max voxel value")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


threshold = 0.8  # Threshold for lesion max voxel value


def remove_lesions_outside_sc(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_rmv_lesions_max_value")
    os.makedirs(temp_folder, exist_ok=True)

    # List of masked segmentation files
    pred_segmentations = [subj_dict['t2_segmentation_file_rmv_lesions_outside_sc']]
    for other_image in subj_dict['other_images']:
        pred_segmentations.append(other_image['segmentation_file_rmv_lesions_outside_sc'])
    
    # In each mask, for each lesion, we check the mask value of the lesion
    for i, pred_seg in enumerate(pred_segmentations):
        ### We enumerate the lesions in the lesion mask by using connected components
        lesion_mask_data = nib.load(pred_seg).get_fdata()
        instances, nb_labels = ndimage.label(lesion_mask_data)
        ### Now we want to split instances into individual masks for each lesion
        individual_instances = np.zeros((nb_labels, *lesion_mask_data.shape), dtype=np.float32)
        for i in range(1, nb_labels+1):
            instance_i = np.zeros_like(lesion_mask_data)
            instance_i[instances == i] = 1
            individual_instances[i-1] = instance_i
        ### For each individual instance, we check the max value
        for i in range(1, nb_labels+1):
            # We select the soft seg of the current lesion
            soft_seg_current_lesion = lesion_mask_data * individual_instances[i-1]
            # binarize the soft seg of the current lesion
            soft_seg_current_lesion_bin = (soft_seg_current_lesion > 0).astype(np.float32)
            # if the max value of the soft seg is below the threshold, we remove the lesion from the mask
            if np.max(soft_seg_current_lesion) < threshold:
                lesion_mask_data = lesion_mask_data * (1 - soft_seg_current_lesion_bin)
        ### Save the modified lesion mask (name : contrast_segmentation_file_rmv_lesions_max_value)
        modified_lesion_mask_path = os.path.join(output_folder, f"{os.path.basename(pred_seg).split('_')[0]}_segmentation_file_rmv_lesions_max_value.nii.gz")
        nib.save(nib.Nifti1Image(lesion_mask_data, nib.load(pred_seg).affine), modified_lesion_mask_path)
        # Update the subject dictionary with the new path
        if i == 0:
            subj_dict['t2_segmentation_file_rmv_lesions_max_value'] = modified_lesion_mask_path
        else:
            subj_dict['other_images'][i-1]['segmentation_file_rmv_lesions_max_value'] = modified_lesion_mask_path

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = remove_lesions_outside_sc(args.subj_dict, args.output_folder)