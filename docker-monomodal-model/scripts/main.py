"""
This is the main script to run inference in the docker.
The steps are the following: 
1. Listing the files
2. Image preprocessing
3. Inference on images
4. Postprocessing of the predicted segmentations
6. Fusion of information
7. Instance segmentation
8. Saving of instance segmentation and csv file with instance probabilities

Args:
    --input_folder: path to the input folder containing the images of only one subject
    --output_folder: path to the output folder

Returns: 
    None

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from listing_inputs import listing_input_files
from image_preprocessing import preprocess_images
from run_inference import run_inference


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

    # Check if SCT works
    print("Checking if SCT is installed and working properly...")
    print("SCT version:")
    assert os.system("sct_version") == 0, "SCT is not installed or not working properly."
    
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
    # Build the path to the model: it is stored in the repo in a folder called "trained-model"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join(parent_dir, "trained-model")
    print(f"Model path: {model_path}")

    # 3. Now we perform inference on the preprocessed images
    subj_dict = run_inference(subj_dict, model_path, temp_folder)

    #########################
    #### POST-PROCESSING ####
    #########################
    # 4. Now we remove lesions outside of the spinal cord

    
     


    return None


if __name__ == "__main__":
    main()