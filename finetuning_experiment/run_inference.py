"""
This script runs the model inferenece on the images in the json dictionnary.
It uses the model of which the path has been passed as an argument.

Input:
    --json: json file containing the images on which to perform inference
    --model: path to the model to use for inference
    --output-dir: path to the output directory where to save the results
    --training: flag to indicate if inference should be ran on training images as well
    --use-gpu: flag to indicate if inference should be ran on GPU or CPU

Output:
    None

Example:
    python run_inference.py --json images_dict.json --model model/patht --output_dir /output_dir/

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import torch
from image import Image
from pathlib import Path

# We define the environment variables here to avoid a warning from nnunetv2
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results']="./nnUNet_results"

# Import for nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a set of images using a trained model.")
    parser.add_argument('--json', type=str, required=True, help='Path to the json file containing the images.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model to use for inference.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory where to save the results.')
    parser.add_argument('--training',  action='store_true', help='Flag to indicate if inference should be ran on training images as well.')
    parser.add_argument('--use-gpu', action='store_true', help='Flag to indicate if inference should be ran on GPU or CPU.')
    return parser.parse_args()


def run_inference(image_input, model_path, output_temp_dir, args):
    """
    This function runs the inference for a single image. 
    It uses the nnUNetPredictor to run the inference and save the results.
    """
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,                      # applies gaussian noise and gaussian blur
        use_mirroring=False,                    # test time augmentation by mirroring on all axes
        device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
        
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )

    # NOTE: for individual files, the image should be in a list of lists
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[image_input]],
        output_folder_or_list_of_truncated_output_files=output_temp_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )


def main(): 
    # Parse the arguments
    args = parse_args()
    json_path = args.json
    model_path = args.model
    output_dir = args.output_dir

    # Build the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the json file
    with open(json_path, 'r') as file:
        images_dict = json.load(file)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    
    # If training is set, we will use the training images as well
    if args.training:
        images = training_images + testing_images
    else:
        images = testing_images

    # Iterate over the images and run inference
    for image in images:
        # For each image input image, we register it to the T2w image
        image_input = image["input_image"]
        t2w_raw_image = image["t2w_raw_image"]
        contrast = image["contrast"]

        # Create a temporary folder to store the results
        temp_folder = Path(output_dir) /  "temp"
        os.makedirs(temp_folder, exist_ok=True)

        # We register the image to the corresponding T2w raw space
        if contrast == 'T2w':
            # If the image is a T2w image then we juste have to move it back to its original space
            assert os.system(f"sct_register_multimodal -i {image_input} -d {t2w_raw_image} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw.nii.gz'} -o {temp_folder/'image_reg.nii.gz'}") == 0
        else:
            # Else we need to compute more complex registration
            parameters = 'step=1,type=im,algo=dl'
            assert os.system(f"sct_register_multimodal -i {image_input} -d {t2w_raw_image} -param {parameters} -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw.nii.gz'} -o {temp_folder/'image_reg.nii.gz'}") == 0
        
        reg_image = os.path.join(temp_folder, "image_reg.nii.gz")

        # Then we reorient the image to RPI
        assert os.system(f"sct_image -i {reg_image} -setorient RPI -o {reg_image}") == 0
        
        temp_folder_pred = os.path.join(output_dir, "temp_pred")
        os.makedirs(temp_folder_pred, exist_ok=True)
        
        # Run the inference
        run_inference(reg_image, model_path, temp_folder_pred, args)

        image_pred = list(Path(temp_folder_pred).rglob("*.nii.gz"))[0]

        # Then we reorient the registration back to the original orientation
        T2w_raw_orientation = Image(t2w_raw_image).orientation
        output_image = os.path.join(output_dir, os.path.basename(image_input))
        assert os.system(f"sct_image -i {image_pred} -setorient {T2w_raw_orientation} -o {output_image}") == 0

        # Remove the temporary folder
        os.system(f"rm -rf {temp_folder}")
        os.system(f"rm -rf {temp_folder_pred}")


if __name__ == "__main__":
    main()