"""
This script merges both prediction by doing a simple average of both predictions.

Author: Pierre-Louis Benveniste
"""
import json
import os
from image import Image, get_dimension
import nibabel as nib
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Merge predictions by averaging both predictions.")
    parser.add_argument("--image_dict", type=str, required=True, help="Path to the JSON file containing image metadata.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where subject folders will be created.")
    return parser.parse_args()


def main():

    args = parse_args()
    image_dict = args.image_dict
    output_folder = args.output_folder

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    # iterate over the images
    for image in tqdm(images):
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        print("Subject name", images[image]["subject_name"])

        # Build path to the lesion masks
        lesion_mask_t2 = os.path.join(sub_folder,"output_rmv_lesion_0.8", "t2w_segmentation_masked_rmvLesion0.8.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder,"output_rmv_lesion_0.8", "psir_segmentation_masked_rmvLesion0.8.nii.gz")

        # Create a temp folder
        temp_folder = os.path.join(sub_folder, "temp_merge")
        os.makedirs(temp_folder, exist_ok=True)

        # We need to segment the spinal cord of the psir inference image
        psir_inference_file = os.path.join(sub_folder, "cropped_reg_psir_preproc_to_t2_raw.nii.gz")
        assert os.system(f"sct_deepseg spinalcord -i {psir_inference_file} -o {os.path.join(temp_folder, 'sc_seg_psir_inference_file.nii.gz')}") == 0
        # We register it to the t2w raw file
        sc_seg_psir_inference_file_reg_to_t2w_raw = os.path.join(temp_folder, 'sc_seg_psir_inference_file_reg_to_t2w_raw.nii.gz')
        assert os.system(f"sct_register_multimodal -i {os.path.join(temp_folder, 'sc_seg_psir_inference_file.nii.gz')} -d {images[image]['t2w_raw_image']} -identity 1 -o {sc_seg_psir_inference_file_reg_to_t2w_raw}") == 0
        # We need to perform the same dilation as the sc seg of the t2 raw file
        ## We get the orientation of the image
        sc_seg_psir_inference_file_reg_to_t2w_raw_orientation = Image(sc_seg_psir_inference_file_reg_to_t2w_raw).orientation
        print("T2w raw image orientation:", sc_seg_psir_inference_file_reg_to_t2w_raw_orientation)
        ## We check which is the S-I direction
        s_i_direction = None
        if 'S' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            s_i_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('S')
        if 'I' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            s_i_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('I')

        # We also want the R-L direction
        r_l_direction = None
        if 'R' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            r_l_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('R')
        if 'L' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            r_l_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('L')
        print("S-I direction:", s_i_direction)
        print("R-L direction:", r_l_direction)

        # Finally we also want the A-P direction
        a_p_direction = None
        if 'A' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            a_p_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('A')
        if 'P' in sc_seg_psir_inference_file_reg_to_t2w_raw_orientation:
            a_p_direction = sc_seg_psir_inference_file_reg_to_t2w_raw_orientation.index('P')
        print("A-P direction:", a_p_direction)

        # Now we want to identify the number of voxels to dilate in the R-L axis (it should be close to 2mm)
        resolution_r_l = get_dimension(Image(sc_seg_psir_inference_file_reg_to_t2w_raw))[4+ r_l_direction]
        print("Resolution R-L:", resolution_r_l)
        vox_dilate_r_l = max(1,int(2 / resolution_r_l))  # 2mm dilation in the R-L axis
        print("Voxels to dilate in R-L axis:", vox_dilate_r_l)

        # same for the A-P axis
        resolution_a_p = get_dimension(Image(sc_seg_psir_inference_file_reg_to_t2w_raw))[4+ a_p_direction]
        print("Resolution A-P:", resolution_a_p)
        vox_dilate_a_p = max(1,int(2 / resolution_a_p)) # 2mm dilation in the A-P axis
        print("Voxels to dilate in A-P axis:", vox_dilate_a_p)

        # We dilate the SC mask around the axial plane (S-I direction) with a radius of 2 mm computed on the R-L axis
        assert os.system(f"sct_maths -i {sc_seg_psir_inference_file_reg_to_t2w_raw} -dilate {vox_dilate_r_l} -shape disk -dim {s_i_direction} -o {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial.nii.gz")==0
        # We dilate the SC mask around the sagittal plane (R-L direction) with a radius of 2 mm computed on the A-P axis
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial.nii.gz -dilate {vox_dilate_a_p} -shape disk -dim {r_l_direction} -o {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz")==0
        # Add the lesion mask to the sc dilated masc
        ## binarize the lesion mask
        assert os.system(f"sct_maths -i {lesion_mask_psir} -bin 0.1 -o {temp_folder}/lesion_mask_psir_bin.nii.gz") == 0
        # We add the lesion mask to the sc segmentation
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz -add {temp_folder}/lesion_mask_psir_bin.nii.gz -o {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz") == 0
        # Binarize the sc segmentation
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz -bin 0.1 -o {sub_folder}/sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz") == 0

        # same for the T2w lesion mask
        assert os.system(f"sct_maths -i {lesion_mask_t2} -bin 0.1 -o {temp_folder}/lesion_mask_t2_bin.nii.gz") == 0
        ## add the lesion mask to the sc segmentation
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz -add {temp_folder}/lesion_mask_t2_bin.nii.gz -o {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz") == 0
        # Binarize the sc segmentation
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz -bin 0.1 -o {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz") == 0

        # Now we add both spinal cord segmentations together
        psir_sc_seg_dilated = os.path.join(sub_folder, "sc_seg_psir_reg_to_t2_raw_dilated_axial_sagittal.nii.gz")
        t2_sc_seg_dilated = os.path.join(sub_folder, "sc_seg_t2_raw_dilated_axial_sagittal.nii.gz")
        assert os.system(f"sct_maths -i {psir_sc_seg_dilated} -add {t2_sc_seg_dilated} -o {sub_folder}/sc_coverage.nii.gz ") == 0

        # We replace all zero values by 1 in sc_coverage file
        sc_coverage = nib.load(os.path.join(sub_folder, "sc_coverage.nii.gz"))
        sc_coverage_data = sc_coverage.get_fdata()
        sc_coverage_data[sc_coverage_data == 0] = 1
        sc_coverage = nib.Nifti1Image(sc_coverage_data, sc_coverage.affine, sc_coverage.header)
        nib.save(sc_coverage, os.path.join(sub_folder, "sc_coverage.nii.gz"))

        # Now we add both lesion masks together
        assert os.system(f"sct_maths -i {lesion_mask_t2} -add {lesion_mask_psir} -o {os.path.join(temp_folder, 'lesion_mask_added.nii.gz')}") == 0
        # We divide the lesion mask by the spinal cord coverage
        assert os.system(f"sct_maths -i {os.path.join(temp_folder, 'lesion_mask_added.nii.gz')} -div {os.path.join(sub_folder, 'sc_coverage.nii.gz')} -o {sub_folder}/lesion_mask_rmv_lesion0p8_merged.nii.gz") == 0

        # Remove the temp folder
        os.system(f"rm -rf {temp_folder}")


if __name__ == "__main__":
    main()
