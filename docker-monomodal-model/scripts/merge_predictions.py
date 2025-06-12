"""
This script merges the lesion masks of a subject accross contrasts

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
    parser = argparse.ArgumentParser(description="Merge lesion masks across contrasts")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def merge_predictions(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_merge_predictions")
    os.makedirs(temp_folder, exist_ok=True)

    # List of masked segmentation files
    inference_files = [subj_dict['t2_inference_file']]
    for other_image in subj_dict['other_images']:
        inference_files.append(other_image['inference_file'])

    # Iterate over the inference files
    for i, inf_file in enumerate(inference_files):
        # We segment the spinal cord and dilate it
        file_sc_seg = os.path.join(temp_folder, f"sc_seg_{i}.nii.gz")
        assert os.system(f"sct_deepseg spinalcord -i {inf_file} -o {file_sc_seg}") == 0
        # We get the orientation of the image
        inf_file_orientation = Image(inf_file).orientation
        # We find which is the S-I direction
        s_i_direction = None
        if 'S' in inf_file_orientation:
            s_i_direction = inf_file_orientation.index('S')
        elif 'I' in inf_file_orientation:
            s_i_direction = inf_file_orientation.index('I')
        # We also want the R-L direction
        r_l_direction = None
        if 'R' in inf_file_orientation:
            r_l_direction = inf_file_orientation.index('R')
        if 'L' in inf_file_orientation:
            r_l_direction = inf_file_orientation.index('L')
        # Finally we also want the A-P direction
        a_p_direction = None
        if 'A' in inf_file_orientation:
            a_p_direction = inf_file_orientation.index('A')
        if 'P' in inf_file_orientation:
            a_p_direction = inf_file_orientation.index('P')
        # Now we want to identify the number of voxels to dilate in the R-L axis (it should be close to 2mm)
        resolution_r_l = get_dimension(Image(inf_file))[4+ r_l_direction]
        vox_dilate_r_l = max(1,int(2 / resolution_r_l))  # 2mm dilation in the R-L axis
        # same for the A-P axis
        resolution_a_p = get_dimension(Image(inf_file))[4+ a_p_direction]
        vox_dilate_a_p = max(1,int(2 / resolution_a_p)) # 2mm dilation in the A-P axis
        # We dilate the SC mask around the axial plane (S-I direction) with a radius of 2 mm computed on the R-L axis
        assert os.system(f"sct_maths -i {file_sc_seg} -dilate {vox_dilate_r_l} -shape disk -dim {s_i_direction} -o {file_sc_seg.replace('.nii.gz', '_dilated_axial.nii.gz')}")==0
        # We dilate the SC mask around the sagittal plane (R-L direction) with a radius of 2 mm computed on the A-P axis
        assert os.system(f"sct_maths -i {file_sc_seg.replace('.nii.gz', '_dilated_axial.nii.gz')} -dilate {vox_dilate_a_p} -shape disk -dim {r_l_direction} -o {file_sc_seg.replace('.nii.gz', '_dilated_axial_sagittal.nii.gz')}")==0
    
    # Now we add all SC segs together
    sc_seg_file_0 = os.path.join(temp_folder, "sc_seg_0_dilated_axial_sagittal.nii.gz")
    sc_seg_file_others = [os.path.join(temp_folder, f"sc_seg_{i}_dilated_axial_sagittal.nii.gz") for i in range(1, len(inference_files))]
    ## We first register the sc segmentation files to the T2w raw file
    assert os.system(f"sct_register_multimodal -i {sc_seg_file_0} -d {subj_dict['t2_raw']} -o {sc_seg_file_0} -identity 1") == 0
    for i, sc_seg_file in enumerate(sc_seg_file_others):
        assert os.system(f"sct_register_multimodal -i {sc_seg_file} -d {subj_dict['t2_raw']} -o {sc_seg_file} -identity 1") == 0
    ## Convert sc_seg_file_others to a string with spaces
    sc_seg_file_others_str = ' '.join(sc_seg_file_others)
    assert os.system(f"sct_maths -i {sc_seg_file_0} -add {sc_seg_file_others_str} -o {temp_folder}/sc_seg_coverage.nii.gz") == 0
    # We replace all zero values by 1 in sc_coverage file
    sc_coverage = nib.load(os.path.join(temp_folder, "sc_seg_coverage.nii.gz"))
    sc_coverage_data = sc_coverage.get_fdata()
    sc_coverage_data[sc_coverage_data == 0] = 1
    sc_coverage = nib.Nifti1Image(sc_coverage_data, sc_coverage.affine, sc_coverage.header)
    nib.save(sc_coverage, os.path.join(temp_folder, "sc_seg_coverage.nii.gz"))

    # Now we add all segmentation files together
    t2_lesion_mask = subj_dict["t2_segmentation_file_rmv_lesions_max_value"]
    other_masks = []
    for other_image in subj_dict['other_images']:
        other_masks.append(other_image['segmentation_file_rmv_lesions_max_value'])
    ## Convert other_masks to a string with spaces
    other_masks_str = ' '.join(other_masks)

    # Now we add both lesion masks together
    assert os.system(f"sct_maths -i {t2_lesion_mask} -add {other_masks_str} -o {os.path.join(temp_folder, 'lesion_mask_added.nii.gz')}") == 0
    # We divide the lesion mask by the spinal cord coverage
    assert os.system(f"sct_maths -i {os.path.join(temp_folder, 'lesion_mask_added.nii.gz')} -div {os.path.join(temp_folder, 'sc_seg_coverage.nii.gz')} -o {os.path.join(temp_folder, 'lesion_mask_merged.nii.gz')}") == 0

    # Move the merged lesion mask to the output folder
    merged_mask_path = os.path.join(output_folder, "merged_lesion_mask.nii.gz")
    assert os.system(f"mv {os.path.join(temp_folder, 'lesion_mask_merged.nii.gz')} {merged_mask_path}") == 0
    
    # Update the subject dictionary with the new path
    subj_dict['merged_lesion_mask'] = merged_mask_path

    # Remove the temp folder
    os.system(f"rm -rf {temp_folder}")

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = merge_predictions(args.subj_dict, args.output_folder)