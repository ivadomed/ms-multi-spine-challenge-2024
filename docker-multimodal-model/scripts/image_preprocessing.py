"""
This script performs the preprocessing: 
    - for now we don't do anything as this will depend on the specific requirements of the model chosen.

Input: 
    -input-images: all the images corresponding to the subject of interest
    -output-folder: the folder where the preprocessed images will be saved

Returns:
    -subj_dict: a dictionary with the paths of the preprocessed images
    -preprocessed-images: the path of the preprocessed images

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import numpy as np 
import nibabel as nib
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input-images", type=str, required=True, help="List of the paths corresponding to the subject images")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()

contrasts = ["T2", "STIR", "PSIR", "MP2RAGE"]

subj_dict = {
    "rawdata_T2": None,
    "rawdata_STIR": None,
    "rawdata_PSIR": None,
    "raw_MP2RAGE": None, 
    "preprocessed_T2": None,
    "preprocessed_STIR": None,
    "preprocessed_PSIR": None,
    "preprocessed_MP2RAGE": None,
}



def preprocess_images(input_images, output_folder):

    

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Build a temp folder in the output folder
    temp_folder = Path(os.path.join(output_folder, "temp_preprocessing"))
    

    os.makedirs(temp_folder, exist_ok=True)

    # We build the subject dictionary
    for image_path in input_images:
        if image_path == None:
            continue
        type = image_path.split('/')[-3]
        contrast = image_path.split('_')[-1].split('.')[0]
        subj_dict[f"{type}_{contrast}"] = image_path

    print("Subject dictionary:", subj_dict)

    # Copy the raw images to the temp folder: we will perform inference on these images
    destination_t2 = os.path.join(output_folder, "MsMultiSpine_001_0000.nii.gz")
    destination_stir = os.path.join(output_folder, "MsMultiSpine_001_0001.nii.gz")
    destination_psir = os.path.join(output_folder, "MsMultiSpine_001_0001.nii.gz")
    destination_mp2rage = os.path.join(output_folder, "MsMultiSpine_001_0001.nii.gz")
    
    if subj_dict["rawdata_T2"] is not None: 
        t2w_raw_image = Path(subj_dict["rawdata_T2"])
    if subj_dict["preprocessed_T2"] is not None:
        file1 = subj_dict["preprocessed_T2"]
        out_image1 = Path(destination_t2)
    if subj_dict["preprocessed_STIR"] is not None:
        file2 = subj_dict["preprocessed_STIR"]
        out_image2 = Path(destination_stir)
    if subj_dict["preprocessed_PSIR"] is not None:
        file2 = subj_dict["preprocessed_PSIR"]
        out_image2 = Path(destination_psir)
    if subj_dict["preprocessed_MP2RAGE"] is not None:
        file2 = subj_dict["preprocessed_MP2RAGE"]
        out_image2 = Path(destination_mp2rage)

    print("Input images:", t2w_raw_image)
    print("Files to preprocess:", file1, file2)

    assert os.system(f"cp {t2w_raw_image} {temp_folder/'raw_t2w.nii.gz'}") == 0
    ## First we need to generate the spinal cord segmentation
    assert os.system(f"sct_deepseg spinalcord -i {temp_folder/'raw_t2w.nii.gz'} -o {temp_folder/'raw_t2w_sc_seg.nii.gz'} ") == 0

    # We register the image to the corresponding T2w raw space
    
    # If the image is a T2w image then we juste have to move it back to its original space
    assert os.system(f"sct_register_multimodal -i {file1} -d {t2w_raw_image} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw1.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw1.nii.gz'}  -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} ") == 0

    # Else we need to compute more complex registration
    parameters = 'step=1,type=im,algo=dl'
    assert os.system(f"sct_register_multimodal -i {file2} -d {t2w_raw_image} -param {parameters} -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw2.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw2.nii.gz'} -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} ") == 0

    # Copy the registered file to the nnunet folder
    assert os.system(f"cp {temp_folder/'image_reg_to_t2wraw1.nii.gz'} {out_image1}") == 0
    assert os.system(f"cp {temp_folder/'image_reg_to_t2wraw2.nii.gz'} {out_image2}") == 0

    
    # Reorient both image and label to desired orientation and location
    assert os.system(f"sct_image -i {out_image1} -setorient RPI -o {out_image1}") == 0
    assert os.system(f"sct_image -i {out_image2} -setorient RPI -o {out_image2}") == 0
    


    
    # Load the image and get the coordinates
    image_data = nib.load(out_image1).get_fdata()
    # Get the cropping box coordinates
    max_idx, min_idx = get_nonzero_bbox(image_data)
    # Crop the images using SCT
    assert os.system(f'sct_crop_image -i {out_image1} -o {out_image1} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
    assert os.system(f'sct_crop_image -i {out_image2} -o {out_image2} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 
    
    

    # We return a dictionnary with the paths of the preprocessed images
    return subj_dict, temp_folder, t2w_raw_image


# Function to extract the coordinates to crop from  
def get_nonzero_bbox(image_data):
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return max_idx, min_idx

if __name__ == "__main__":
    args = parse_args()
    subj_dict, preprocessed_images = preprocess_images(args.input_images, args.output_folder)