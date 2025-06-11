"""
This script performs post-processing on the results of experiment 251.
It removes lesions outside of the spinal cord. 
"""
import json
import os
from image import Image, get_dimension

def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_151_prep"

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

        # Segment the spinal cord of the T2w raw image
        assert os.system(f"sct_deepseg spinalcord -i {images[image]['t2w_raw_image']} -o {sub_folder}/sc_seg_t2_raw.nii.gz ") == 0

        # Build path to sc seg of the t2w raw image
        sc_seg_t2_raw = os.path.join(sub_folder, "sc_seg_t2_raw.nii.gz")

        # Build path to the lesion masks
        lesion_mask_t2 = os.path.join(sub_folder,"predictions", "t2w_segmentation.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder,"predictions", "psir_segmentation.nii.gz")

        # We get the orientation of the image
        t2w_raw_image_orientation = Image(images[image]['t2w_raw_image']).orientation
        print("T2w raw image orientation:", t2w_raw_image_orientation)

        # We check which is the S-I direction
        s_i_direction = None
        if 'S' in t2w_raw_image_orientation:
            s_i_direction = t2w_raw_image_orientation.index('S')
        if 'I' in t2w_raw_image_orientation:
            s_i_direction = t2w_raw_image_orientation.index('I')

        # We also want the R-L direction
        r_l_direction = None
        if 'R' in t2w_raw_image_orientation:
            r_l_direction = t2w_raw_image_orientation.index('R')
        if 'L' in t2w_raw_image_orientation:
            r_l_direction = t2w_raw_image_orientation.index('L')
        print("S-I direction:", s_i_direction)
        print("R-L direction:", r_l_direction)

        # Finally we also want the A-P direction
        a_p_direction = None
        if 'A' in t2w_raw_image_orientation:
            a_p_direction = t2w_raw_image_orientation.index('A')
        if 'P' in t2w_raw_image_orientation:
            a_p_direction = t2w_raw_image_orientation.index('P')
        print("A-P direction:", a_p_direction)

        # Now we want to identify the number of voxels to dilate in the R-L axis (it should be close to 2mm)
        resolution_r_l = get_dimension(Image(images[image]['t2w_raw_image']))[4+ r_l_direction]
        print("Resolution R-L:", resolution_r_l)
        vox_dilate_r_l = max(1,int(2 / resolution_r_l))  # 2mm dilation in the R-L axis
        print("Voxels to dilate in R-L axis:", vox_dilate_r_l)

        # same for the A-P axis
        resolution_a_p = get_dimension(Image(images[image]['t2w_raw_image']))[4+ a_p_direction]
        print("Resolution A-P:", resolution_a_p)
        vox_dilate_a_p = max(1,int(2 / resolution_a_p)) # 2mm dilation in the A-P axis
        print("Voxels to dilate in A-P axis:", vox_dilate_a_p)

        # We dilate the SC mask around the axial plane (S-I direction) with a radius of 2 mm computed on the R-L axis
        assert os.system(f"sct_maths -i {sc_seg_t2_raw} -dilate {vox_dilate_r_l} -shape disk -dim {s_i_direction} -o {sub_folder}/sc_seg_t2_raw_dilated_axial.nii.gz")==0
        # We dilate the SC mask around the sagittal plane (R-L direction) with a radius of 2 mm computed on the A-P axis
        assert os.system(f"sct_maths -i {sub_folder}/sc_seg_t2_raw_dilated_axial.nii.gz -dilate {vox_dilate_a_p} -shape disk -dim {r_l_direction} -o {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz")==0

        # # We remove the lesions outside of the spinal cord
        assert os.system(f"sct_maths -i {lesion_mask_t2} -mul {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz -o {lesion_mask_t2.replace('.nii.gz', '_masked.nii.gz')}")==0
        assert os.system(f"sct_maths -i {lesion_mask_psir} -mul {sub_folder}/sc_seg_t2_raw_dilated_axial_sagittal.nii.gz -o {lesion_mask_psir.replace('.nii.gz', '_masked.nii.gz')}")==0


if __name__ == "__main__":
    main()