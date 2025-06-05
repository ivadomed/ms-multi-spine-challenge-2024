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

# We define the environment variables here to avoid a warning from nnunetv2
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results']="./nnUNet_results"

# Import for nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model folder")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def run_inference(input_image, model_path, output_folder):

    # Build a temp folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_inference")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Initialize the model 
    predictor = nnUNetPredictor(
        tile_step_size=0.5,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,                      # applies gaussian noise and gaussian blur
        use_mirroring=True,                    # test time augmentation by mirroring on all axes
        device=torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
        
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=[0],
        checkpoint_name='checkpoint_best.pth',
    )

    # Run inference on the input image
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[input_image]],
        output_folder_or_list_of_truncated_output_files=temp_folder,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    # Build output image path (output folder and image name)
    output_image = os.path.join(temp_folder, Path(input_image).name)

    return output_image


if __name__ == "__main__":
    args = parse_args()
    output_image = run_inference(args.input_image, args.output_folder)