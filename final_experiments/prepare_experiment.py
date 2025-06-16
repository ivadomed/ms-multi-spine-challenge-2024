"""
In this script, we prepare the experiment (for monomodal 151) by creating folders for each subject with the necessary files for evaluation. This way, we aim at increase the speed of testing. 

"""
import json
import os
import nibabel as nib
import numpy as np
import argparse


def get_nonzero_bbox(image_data):
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return max_idx, min_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare experiment 151 by creating subject folders with necessary files.")
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
    for image in images:
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        os.makedirs(sub_folder, exist_ok=True)
        # Copy the T2 raw file
        assert os.system(f"cp {images[image]['t2w_raw_image']} {sub_folder}") == 0
        # Copy the psir raw file
        ## Copy the PSIR preprocessed image
        assert os.system(f"cp {images[image]['input_image']} {sub_folder}") == 0
        ## Build the file path:
        psir_raw_image = images[image]["input_image"].replace("desc-preproc_", "")
        assert os.system(f"cp {psir_raw_image} {sub_folder}") == 0
        # Copy the T2w raw mask
        assert os.system(f"cp {images[image]['t2w_raw_label_file']} {sub_folder}") == 0

        # Build a temp folder
        temp_folder = os.path.join(sub_folder, "temp")
        os.makedirs(temp_folder, exist_ok=True)

        ## We register the psir preproc to the T2w raw image to build a warping field
        parameters = 'step=1,type=im,algo=dl'
        assert os.system(f"sct_register_multimodal -i {images[image]['input_image']} -d {images[image]['t2w_raw_image']} -param {parameters} -o {sub_folder}/reg_psir_preproc_to_t2_raw.nii.gz") == 0
        ### To match orientation and resolution, we register the image to the T2w raw image
        assert os.system(f"sct_register_multimodal -i {sub_folder}/reg_psir_preproc_to_t2_raw.nii.gz -d {images[image]['t2w_raw_image']} -identity 1 -o {sub_folder}/reg_psir_preproc_to_t2_raw.nii.gz") == 0
        ### Then we use the reg_image and remove the empty space
        reg_image = os.path.join(sub_folder, "reg_psir_preproc_to_t2_raw.nii.gz")
        reg_image_cropped = os.path.join(sub_folder, "cropped_reg_psir_preproc_to_t2_raw.nii.gz")
        reg_image_data = nib.load(reg_image).get_fdata()
        ### Get the cropping box coordinates
        max_idx, min_idx = get_nonzero_bbox(reg_image_data)
        assert os.system(f'sct_crop_image -i {reg_image} -o {reg_image_cropped} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
        
        ## Same for the T2w preproc image
        t2w_preproc_image = images[image]["t2w_raw_image"].replace("_T2w.","_desc-preproc_T2w.")
        assert os.system(f"sct_register_multimodal -i {t2w_preproc_image} -d {images[image]['t2w_raw_image']} -identity 1 -o {sub_folder}/reg_t2w_preproc_to_t2w_raw.nii.gz") == 0
        # Then we use the reg_image and remove the empty space
        reg_image = os.path.join(sub_folder, "reg_t2w_preproc_to_t2w_raw.nii.gz")
        reg_image_cropped = os.path.join(sub_folder, "cropped_reg_t2w_preproc_to_t2w_raw.nii.gz")
        reg_image_data = nib.load(reg_image).get_fdata()
        # Get the cropping box coordinates
        max_idx, min_idx = get_nonzero_bbox(reg_image_data)
        assert os.system(f'sct_crop_image -i {reg_image} -o {reg_image_cropped} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0

        # Clean up the temp folder
        assert os.system(f"rm -rf {temp_folder}") == 0
        # Clean useless files
        assert os.system(f"rm -rf {sub_folder}/warp_*") == 0
        # Remove reg files
        assert os.system(f"rm -rf {sub_folder}/reg_*") == 0

    print("done for all")


if __name__ == "__main__":
    main()