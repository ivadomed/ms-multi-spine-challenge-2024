"""
This script runs the inference on the images using the 5 folds of the model.

Input: 
    -input_image: path to the input image
    -output_folder: path to the output folder

Returns: 
    -output_image: path to the output image

Author: Pierre-Louis Benveniste     
"""
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def run_inference(input_image, output_folder):

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_inference")
    os.makedirs(temp_folder, exist_ok=True)

    # Inference running here
    #TODO

    # Build output image path (output folder and image name)
    output_image = os.path.join(temp_folder, Path(input_image).name)

    return output_image


if __name__ == "__main__":
    args = parse_args()
    output_image = run_inference(args.input_image, args.output_folder)