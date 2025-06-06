"""
This script runs the inference on the images using the 5 folds of the model.

Input: 
    -input_image: path to the input image
    -model_path: path to the model folder
    -output_folder: path to the output folder

Returns: 
    -output_image: path to the output image

Author: Pierre-Louis Benveniste     
"""
import argparse
import os
from pathlib import Path
import torch



# Import for nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input folder")
    #parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model folder")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def run_inference(input_image, output_folder):

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_inference")
    os.makedirs(temp_folder, exist_ok=True)
    
    for fold in range(5):
        # Run inference for the fold
        assert os.system(f"nnUNetv2_predict -i {input_image} -o {output_folder}/fold_{fold}/prediction -d 200 -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDA5_150epochs -c 2d -f {fold} -chk checkpoint_best_{fold}.pth ") == 0
    
    return output_folder


if __name__ == "__main__":
    args = parse_args()
    output_image = run_inference(args.input_image, args.output_folder)