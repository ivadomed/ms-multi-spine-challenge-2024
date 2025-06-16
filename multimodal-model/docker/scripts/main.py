"""
This is the main script to run inference in the docker.
The steps are the following: 
1. Listing the files
2. Image preprocessing
3. Inference on images
4. Postprocessing of the predicted segmentations
5. Instance segmentation
6. Saving of instance segmentation and csv file with instance probabilities

Args:
    --input_folder: path to the input folder containing the images of only one subject
    --output_folder: path to the output folder

Returns: 
    None

Author: Thomas Dagonneau & Pierre-Louis Benveniste
"""
import argparse
import os
from listing_inputs import listing_input_files
from image_preprocessing import preprocess_images
from run_inference import run_inference
from remove_lesions_outside_sc import remove_lesions_outside_sc
from remove_lesions_max_value import remove_lesions_max_value
from remove_small_lesions import remove_small_lesions
from format_segmentation import format_segmentation
from pred_postprocessing import postprocess_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def main():

    # Parse the arguments
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder in the output folder
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    #######################
    #### PREPROCESSING ####
    #######################

    # 1. List all the files in the input folder 
    list_files = listing_input_files(input_folder)

    # 2. Now we perform image preprocessing
    subj_dict = preprocess_images(list_files, temp_folder)
   
    ###################
    #### INFERENCE ####
    ###################
    
    # 3. Now we perform inference on the preprocessed images
    
    print(f"Running inference...")
    subj_dict = run_inference(subj_dict, temp_folder)

    #########################
    #### POST-PROCESSING ####
    #########################

    # 4. Postprocess the segmentations
    subj_dict = postprocess_segmentation(subj_dict)
    print(subj_dict)

    #  Now we remove lesions outside of the spinal cord
    subj_dict = remove_lesions_outside_sc(subj_dict, temp_folder)

    #  Now we remove lesions where max voxel value is below 0.3
    subj_dict = remove_lesions_max_value(subj_dict, temp_folder)
    
    # Remove small lesions  (less than 18 voxels)
    subj_dict = remove_small_lesions(subj_dict, temp_folder)

    ############################
    #### FORMAT PREDICTIONS ####
    ############################
    # 9. Format the prediction
    subj_dict = format_segmentation(subj_dict, output_folder)

    # Remove the temporary folder
    assert os.system(f"rm -rf {temp_folder}") == 0, "Failed to remove the temporary folder."

    return None


if __name__ == "__main__":
    main()