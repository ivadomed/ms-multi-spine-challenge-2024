"""
This script binarizes the lesion mask with a threshold of 0.8.

Input: 
    -subj_dict: the dictionnary of the subject
    -output_folder: the output folder where the segmentation masks will be saved

Output:
    -subj_dict: the updated dictionnary of the subject with the segmentation masks updated

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from image import Image, get_dimension
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Binarize lesion mask script")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


thresh = 0.8  # Threshold to binarize the lesion mask


def binarize_prediction(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Build the output path
    os.makedirs(output_folder, exist_ok=True)

    # Build input and output
    merged_lesion_mask = subj_dict['merged_lesion_mask']
    binarized_lesion_mask = os.path.join(output_folder, "binarized_lesion_mask.nii.gz")
    # Binarize the lesion mask
    assert os.system(f"sct_maths -i {merged_lesion_mask} -bin {thresh} -o {binarized_lesion_mask} ") == 0

    # Update the subject dictionary with the new path
    subj_dict['binarized_lesion_mask'] = binarized_lesion_mask

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = binarize_prediction(args.subj_dict, args.output_folder)