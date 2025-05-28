"""
This script runs the model inferenece on the images in the json dictionnary.
It uses the model of which the path has been passed as an argument.

Input:
    --json: json file containing the images on which to perform inference
    --model: path to the model to use for inference
    --output-dir: path to the output directory where to save the results
    --training: flag to indicate if inference should be ran on training images as well
    --use-gpu: flag to indicate if inference should be ran on GPU or CPU
    --use-checkpoint-best: flag to indicate if inference should be done with the "Checkpoint_best.pth" model (by default it uses Checkpoint_final.pth)

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
import numpy as np
import nibabel as nib


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
    parser.add_argument('--use-checkpoint-best', action='store_true', help='Flag to indicate if inference should be done with the "Checkpoint_best.pth" model (by default it uses Checkpoint_final.pth).')
    return parser.parse_args()


# Function to extract the coordinates to crop from  
def get_nonzero_bbox(image_data):
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return max_idx, min_idx


def run_inference(image_input, predictor, output_temp_dir):
    """
    This function runs the inference for a single image. 
    It uses the nnUNetPredictor to run the inference and save the results.
    """

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

    return None


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

    # Initialize the model 
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
        checkpoint_name='checkpoint_best.pth' if args.use_checkpoint_best else 'checkpoint_final.pth',
    )

    # Iterate over the images and run inference
    for image in images:
        # For each image input image, we register it to the T2w image
        image_input = images[image]["input_image"] # this is the image we want to run inference on: it is a preprocessed image (for SC cropping)
        t2w_raw_image = images[image]["t2w_raw_image"]
        contrast = images[image]["contrast"]
        image_raw = image_input.replace('_desc-preproc', '')  # This is the raw image, without the preproc suffix

        # Create a temporary folder to store the results
        temp_folder = Path(output_dir) /  "temp"
        os.makedirs(temp_folder, exist_ok=True)

        # We segment on the preproc image, which we need in the the image raw space
        assert os.system(f"sct_register_multimodal -i {image_input} -d {image_raw} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_image_raw.nii.gz'} -o {temp_folder/'image_reg_image_raw.nii.gz'}") == 0

        # Then we use the reg_image and remove the empty space
        reg_image = os.path.join(temp_folder, "image_reg_image_raw.nii.gz")
        reg_image_cropped = os.path.join(temp_folder, "image_reg_image_raw_cropped.nii.gz")
        reg_image_data = nib.load(reg_image).get_fdata()
        # Get the cropping box coordinates
        max_idx, min_idx = get_nonzero_bbox(reg_image_data)
        # Crop the image to the bounding box
        assert os.system(f'sct_crop_image -i {reg_image} -o {reg_image_cropped} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
        
        # Then we reorient the image to RPI
        reg_image_cropped_rpi = os.path.join(temp_folder, "image_reg_image_raw_cropped_rpi.nii.gz")
        assert os.system(f"sct_image -i {reg_image_cropped} -setorient RPI -o {reg_image_cropped_rpi}") == 0

        temp_folder_pred = os.path.join(output_dir, "temp_pred")
        os.makedirs(temp_folder_pred, exist_ok=True)
        
        # Run the inference
        run_inference(reg_image_cropped_rpi, predictor, temp_folder_pred)

        image_pred = list(Path(temp_folder_pred).rglob("*.nii.gz"))[0]

        # Then we reorient the registration back to the original orientation
        image_raw_orientation = Image(image_raw).orientation
        pred_reg_back_to_raw = os.path.join(temp_folder, "pred_reg_back_to_raw.nii.gz")
        assert os.system(f"sct_image -i {image_pred} -setorient {image_raw_orientation} -o {pred_reg_back_to_raw}") == 0
        # We register the image back to the original image space
        assert os.system(f"sct_register_multimodal -i {pred_reg_back_to_raw} -d {image_raw} -identity 1 -ofolder {temp_folder} -o {temp_folder/'pred_reg_image_raw.nii.gz'}") == 0
        
        output_label = os.path.join(output_dir, os.path.basename(image_input))

        # If the image is in the T2w raw space, we are done
        if contrast == "t2w_raw":
            assert os.system(f"cp {temp_folder/'pred_reg_image_raw.nii.gz'} {output_label}") == 0
        # Else we need to register the predicted segmentation to the T2w raw image
        else :  
            # We build a warping field from image raw to T2_raw
            # 0.1 First we need to generate the spinal cord segmentation
            assert os.system(f"sct_deepseg -i {image_raw} -o {temp_folder/'image_raw_image_sc_seg.nii.gz'} -task seg_sc_contrast_agnostic ") == 0
            # 0.2 Then we need to generate the spinal cord segmentation of the T2w raw image
            assert os.system(f"sct_deepseg -i {t2w_raw_image} -o {temp_folder/'t2_raw_image_sc_seg.nii.gz'} -task seg_sc_contrast_agnostic ") == 0
            # 1. We register the image raw to the T2w raw image to build a warping field
            parameters = 'step=1,type=im,algo=dl,metric=MI:step=2,type=seg,algo=syn,metric=MeanSquares:step=3,type=im,algo=syn,metric=MI,iter=5,shrink=2'
            assert os.system(f"sct_register_multimodal -i {image_raw} -d {t2w_raw_image} -param {parameters} -ofolder {temp_folder} -owarp {temp_folder/'warp_image_raw_to_t2_raw.nii.gz'} -o {temp_folder/'imageraw_reg_to_t2raw.nii.gz'} -iseg {temp_folder/'image_raw_image_sc_seg.nii.gz'} -dseg {temp_folder/'t2_raw_image_sc_seg.nii.gz'} ") == 0
            # 2. We apply the warping field to the predicted segmentation
            assert os.system(f"sct_apply_transfo -i {temp_folder/'pred_reg_image_raw.nii.gz'} -d {t2w_raw_image} -w {temp_folder/'warp_image_raw_to_t2_raw.nii.gz'} -o {output_label} -x linear") == 0

        # Remove the temporary folder
        os.system(f"rm -rf {temp_folder}")
        os.system(f"rm -rf {temp_folder_pred}")


if __name__ == "__main__":
    main()