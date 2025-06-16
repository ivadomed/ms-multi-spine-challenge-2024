"""
This script performs the preprocessing: 
    - preproc images are all registered to the T2w raw image
    - images are cropped to the non-zero bounding box of the registered T2w preprocessed image

Input: 
    -input-images: all the images corresponding to the subject of interest
    -output-folder: the folder where the preprocessed images will be saved

Returns:
    -subj_dict: a dictionary with the paths of the preprocessed images and their respective inference files

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input-images", type=str, required=True, help="List of the paths corresponding to the subject images")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def get_nonzero_bbox(image_data):
    """
    This function computes the bounding box of the non-zero regions in a 3D image.
    """
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return max_idx, min_idx


contrasts = ["T2", "STIR", "PSIR", "MP2RAGE"]


def preprocess_images(input_images, output_folder):

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_preprocessing")
    os.makedirs(temp_folder, exist_ok=True)

    # Check that the T2w image is present in the input images (i.e. the first image in the list)
    assert input_images[0] is not None, "The T2w image is not present in the input images."

    # Initialize the subject dictionary
    subj_dict = {}
    subj_dict["t2_raw"] = input_images[0]
    subj_dict["t2_preproc"] = input_images[4]
    subj_dict["t2_inference_file"] = None
    other_contrasts_images = []

    # Iterate over the other three contrasts
    for i in range(1, 4):
        if input_images[i] is not None:
            new_image = {
                "image_raw": input_images[i],
                "image_preproc": input_images[i + 4],
                "contrast": contrasts[i],
                "inference_file": None
            }
            other_contrasts_images.append(new_image)
    subj_dict["other_images"] = other_contrasts_images

    # We first preprocess the T2w image
    ## We start by copying the images in the temp folder
    assert os.system(f"cp {subj_dict['t2_raw']} {temp_folder}/t2w_raw_image.nii.gz") == 0
    assert os.system(f"cp {subj_dict['t2_preproc']} {temp_folder}/t2w_preproc_image.nii.gz") == 0
    ## We register the T2w preprocessed image to the T2w raw image
    assert os.system(f"sct_register_multimodal -i {temp_folder}/t2w_preproc_image.nii.gz -d {temp_folder}/t2w_raw_image.nii.gz -identity 1 -o {temp_folder}/reg_t2w_preproc_to_t2w_raw.nii.gz") == 0
    ## Then we remove the empty space in the registered image
    ### Build the paths
    reg_image = os.path.join(temp_folder, "reg_t2w_preproc_to_t2w_raw.nii.gz")
    reg_image_cropped = os.path.join(temp_folder, "cropped_reg_t2w_preproc_to_t2w_raw.nii.gz")
    reg_image_data = nib.load(reg_image).get_fdata()
    ### Get the cropping box coordinates and crop the image
    max_idx, min_idx = get_nonzero_bbox(reg_image_data)
    assert os.system(f'sct_crop_image -i {reg_image} -o {reg_image_cropped} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
    ## We copy the inference file to the output dictionary
    t2_inference_file = os.path.join(output_folder, "t2w_inference.nii.gz")
    assert os.system(f"cp {reg_image_cropped} {t2_inference_file}") == 0
    ## We update the subject dictionary with the T2w inference file
    subj_dict["t2_inference_file"] = t2_inference_file

    # We now preprocess the other contrasts
    for other_image in subj_dict["other_images"]:
        ## We start by copying the image to the image to the temp folder
        assert os.system(f"cp {other_image['image_preproc']} {temp_folder}/{other_image['contrast']}_preproc_image.nii.gz") == 0
        ## We register the preprocessed image to the T2w raw image
        parameters = 'step=1,type=im,algo=dl'
        assert os.system(f"sct_register_multimodal -i {temp_folder}/{other_image['contrast']}_preproc_image.nii.gz -d {temp_folder}/t2w_raw_image.nii.gz -param {parameters} -o {temp_folder}/reg_{other_image['contrast']}_preproc_to_t2w_raw.nii.gz") == 0
        ## To match orientation and resolution, we register the image to the T2w raw image
        assert os.system(f"sct_register_multimodal -i {temp_folder}/reg_{other_image['contrast']}_preproc_to_t2w_raw.nii.gz -d {temp_folder}/t2w_raw_image.nii.gz -identity 1 -o {temp_folder}/reg_{other_image['contrast']}_preproc_to_t2w_raw.nii.gz") == 0
        ## Then we remove the empty space in the registered image
        reg_image = os.path.join(temp_folder, f"reg_{other_image['contrast']}_preproc_to_t2w_raw.nii.gz")
        reg_image_cropped = os.path.join(temp_folder, f"cropped_reg_{other_image['contrast']}_preproc_to_t2w_raw.nii.gz")
        reg_image_data = nib.load(reg_image).get_fdata()
        ### Get the cropping box coordinates and crop the image
        max_idx, min_idx = get_nonzero_bbox(reg_image_data)
        assert os.system(f'sct_crop_image -i {reg_image} -o {reg_image_cropped} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
        ## We copy the inference file to the output dictionary
        other_image_inference_file = os.path.join(output_folder, f"{other_image['contrast']}_inference.nii.gz")
        assert os.system(f"cp {reg_image_cropped} {other_image_inference_file}") == 0
        ## We update the subject dictionary with the inference file
        subj_dict["other_images"][subj_dict["other_images"].index(other_image)]["inference_file"] = other_image_inference_file
    
    # We remove the temp folder
    # assert os.system(f"rm -rf {temp_folder}") == 0

    # Return the subject dictionary
    return subj_dict


# if __name__ == "__main__":
#     args = parse_args()
#     subj_dict = preprocess_images(args.input_images, args.output_folder)