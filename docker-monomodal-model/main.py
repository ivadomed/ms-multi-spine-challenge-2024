"""
This is the main script to run inference in the docker.
The steps are the following: 
1. Listing the files
2. Image preprocessing
2. Inference on images
3. Fusion of information
4. Calibration
5. Instance segmentation
6. Post-processing for output

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

    # List all the files in the input folder 
    list_files = listing_input_files(input_folder)

    # Now we perform image preprocessing
    subj_dict, preprocessed_images = preprocess_images(list_files, output_folder)
    
    # Now we perform inference on the preprocessed images
    





    return None


if __name__ == "__main__":
    main()