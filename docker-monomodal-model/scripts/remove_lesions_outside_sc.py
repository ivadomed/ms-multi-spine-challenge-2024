"""
This script removes lesions outside the spinal cord from the segmentation masks.
To do so, it segments the spinal cord in the T2w raw image.
The spinal cord segmentation is dilated by about 2mm on the axial plane. 
The lesions outside the spinal cord mask are then removed from the segmentation masks.

Input: 
    -subj_dict: the dictionnary of the subject
    -output_folder: the output folder where the segmentation masks will be saved

Output:
    -subj_dict: the updated dictionnary of the subject with the segmentation masks updated

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from image import Image, get_dimension


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions outside spinal cord script")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def remove_lesions_outside_sc(subj_dict, output_folder):
    """
    This is the main function of the script
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_rmv_lesions_outside_sc")
    os.makedirs(temp_folder, exist_ok=True)

    # List of inference files
    pred_segmentations = [subj_dict['t2_segmentation_file']]
    for other_image in subj_dict['other_images']:
        pred_segmentations.append(other_image['segmentation_file'])
    
    # Segment the spinal cord of the T2w raw image
    assert os.system(f"sct_deepseg spinalcord -i {subj_dict['t2_raw']} -o {temp_folder}/sc_seg_t2_raw.nii.gz ") == 0
    # Build the path to the spinal cord segmentation
    sc_seg_t2_raw = os.path.join(temp_folder, "sc_seg_t2_raw.nii.gz")

    # We get the orientation of the image
    t2w_raw_image_orientation = Image(subj_dict['t2_raw']).orientation
    print("T2w raw image orientation:", t2w_raw_image_orientation)

    # We find which is the S-I direction
    s_i_direction = None
    if 'S' in t2w_raw_image_orientation:
        s_i_direction = t2w_raw_image_orientation.index('S')
    elif 'I' in t2w_raw_image_orientation:
        s_i_direction = t2w_raw_image_orientation.index('I')

    # We also want the R-L direction
    r_l_direction = None
    if 'R' in t2w_raw_image_orientation:
        r_l_direction = t2w_raw_image_orientation.index('R')
    if 'L' in t2w_raw_image_orientation:
        r_l_direction = t2w_raw_image_orientation.index('L')

    # Finally we also want the A-P direction
    a_p_direction = None
    if 'A' in t2w_raw_image_orientation:
        a_p_direction = t2w_raw_image_orientation.index('A')
    if 'P' in t2w_raw_image_orientation:
        a_p_direction = t2w_raw_image_orientation.index('P')
    

    # Now we want to identify the number of voxels to dilate in the R-L axis (it should be close to 2mm)
    resolution_r_l = get_dimension(Image(subj_dict['t2_raw']))[4+ r_l_direction]
    vox_dilate_r_l = max(1,int(2 / resolution_r_l))  # 2mm dilation in the R-L axis

    # same for the A-P axis
    resolution_a_p = get_dimension(Image(subj_dict['t2_raw']))[4+ a_p_direction]
    vox_dilate_a_p = max(1,int(2 / resolution_a_p)) # 2mm dilation in the A-P axis

    # We dilate the SC mask around the axial plane (S-I direction) with a radius of 2 mm computed on the R-L axis
    assert os.system(f"sct_maths -i {sc_seg_t2_raw} -dilate {vox_dilate_r_l} -shape disk -dim {s_i_direction} -o {temp_folder}/sc_seg_t2_raw_dilated_axial.nii.gz")==0
    # We dilate the SC mask around the sagittal plane (R-L direction) with a radius of 2 mm computed on the A-P axis
    assert os.system(f"sct_maths -i {temp_folder}/sc_seg_t2_raw_dilated_axial.nii.gz -dilate {vox_dilate_a_p} -shape disk -dim {r_l_direction} -o {temp_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz")==0

    # For all segmentation files, we remove the lesions outside of the spinal cord
    for seg_file in pred_segmentations:
        # Build the output path
        output_seg_path = os.path.join(output_folder, Path(seg_file).name.replace('.nii.gz', '_rmv_lesions_outside_sc.nii.gz'))
        assert os.system(f"sct_maths -i {seg_file} -mul {temp_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz -o {output_seg_path}")==0
    
    # Update the subject dictionary with the new segmentation paths
    subj_dict['t2_segmentation_file_rmv_lesions_outside_sc'] = subj_dict["t2_segmentation_file"].replace('.nii.gz', '_rmv_lesions_outside_sc.nii.gz')
    for other_image in subj_dict['other_images']:
        other_image['segmentation_file_rmv_lesions_outside_sc'] = other_image['segmentation_file'].replace('.nii.gz', '_rmv_lesions_outside_sc.nii.gz')

    # Remove the temporary folder
    assert os.system(f"rm -rf {temp_folder}") == 0

    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    updated_subj_dict = remove_lesions_outside_sc(args.subj_dict, args.output_folder)