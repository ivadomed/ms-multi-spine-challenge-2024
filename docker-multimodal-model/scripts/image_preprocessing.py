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

    print(input_images)
    print(output_folder)

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_preprocessing")
    os.makedirs(temp_folder, exist_ok=True)

    # We build the subject dictionary
    for image_path in input_images:
        if image_path == None:
            continue
        type = image_path.split('/')[-3]
        contrast = image_path.split('_')[-1].split('.')[0]
        subj_dict[f"{type}_{contrast}"] = image_path

    # Copy the raw images to the temp folder: we will perform inference on these images
    destination_t2 = os.path.join(temp_folder, "T2.nii.gz")
    destination_stir = os.path.join(temp_folder, "STIR.nii.gz")
    destination_psir = os.path.join(temp_folder, "PSIR.nii.gz")
    destination_mp2rage = os.path.join(temp_folder, "MP2RAGE.nii.gz")
    preprocessed_images = {}
    if subj_dict["rawdata_T2"] is not None:
        assert os.system(f"sct_image -i {subj_dict['rawdata_T2']} -setorient RPI -o {destination_t2}") == 0
        #os.system(f"cp {subj_dict['rawdata_T2']} {destination_t2}")
        preprocessed_images["T2"] = destination_t2
    if subj_dict["rawdata_STIR"] is not None:
        os.system(f"cp {subj_dict['rawdata_STIR']} {destination_stir}")
        preprocessed_images["STIR"] = destination_stir
    if subj_dict["rawdata_PSIR"] is not None:
        os.system(f"cp {subj_dict['rawdata_PSIR']} {destination_psir}")
        preprocessed_images["PSIR"] = destination_psir
    if subj_dict["raw_MP2RAGE"] is not None:
        os.system(f"cp {subj_dict['raw_MP2RAGE']} {destination_mp2rage}")
        preprocessed_images["MP2RAGE"] = destination_mp2rage

    # We return a dictionnary with the paths of the preprocessed images
    return subj_dict, preprocessed_images


if __name__ == "__main__":
    args = parse_args()
    subj_dict, preprocessed_images = preprocess_images(args.input_images, args.output_folder)