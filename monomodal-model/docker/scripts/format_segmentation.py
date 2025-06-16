"""
This script formats the segmentation to an instance segmentation. 
It transforms the binary segmentation into an instance segmentation. 
It also produces an json file with the probability scores associated to the lesions

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
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Format segmentation script")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def format_segmentation(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Build the output path
    os.makedirs(output_folder, exist_ok=True)

    # Create a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_format")

    # Build input mask path
    lesion_mask_bin = subj_dict['lesion_mask_rmv_small_lesion']
    lesion_mask_soft = subj_dict['merged_lesion_mask']

    # We multiply the soft segmentation mask by the binary mask to keep only the lesions
    lesion_mask_soft_data = nib.load(lesion_mask_soft).get_fdata()
    lesion_mask_bin_data = nib.load(lesion_mask_bin).get_fdata()
    final_lesion_mask_soft_data = lesion_mask_soft_data * lesion_mask_bin_data

    # Now we make an instance segmentation from the soft mask
    instances, nb_labels = ndimage.label(final_lesion_mask_soft_data)

    # Initialize the list to store lesion probabilities
    lesion_probabilities = []

    ## For each instance, we want to extract the soft segmentation mask of the lesion
    individual_instances = np.zeros((nb_labels, *final_lesion_mask_soft_data.shape), dtype=np.float32)
    for i in range(1, nb_labels + 1):
        instance_i = np.zeros_like(final_lesion_mask_soft_data)
        instance_i[instances == i] = 1
        individual_instances[i - 1] = instance_i * final_lesion_mask_soft_data
    ## for each individual instance, we compute the lesion probability
    for i in range(1, nb_labels + 1):
        # Compute the lesion probability as the mean value of the soft segmentation mask
        lesion_probability = np.mean(individual_instances[i - 1][individual_instances[i - 1] > 0])
        lesion_probabilities.append({
            "lesion_id": i,
            "lesion_probability": float(lesion_probability)
        })

    # Save the instance segmentation as a NIfTI file
    instance_segmentation_path = os.path.join(output_folder, "instance_segmentation.nii.gz")
    nib.save(nib.Nifti1Image(instances, nib.load(lesion_mask_soft).affine), instance_segmentation_path)
    
    # Lesion probability should be save in a csv file with a column id and a column p
    output_csv_path = os.path.join(output_folder, "lesion_probabilities.csv")
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'p'])
        for lesion in lesion_probabilities:
            writer.writerow([lesion['lesion_id'], lesion['lesion_probability']])

    # Update the subject dictionary with the new paths
    subj_dict['instance_segmentation_file'] = instance_segmentation_path
    subj_dict['lesion_probabilities_file'] = output_csv_path

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = format_segmentation(args.subj_dict, args.output_folder)