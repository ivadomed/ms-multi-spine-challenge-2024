"""
This script runs the inference on the images using the 5 folds of the model.

Input: 
    -subj-dict: path to the subject dictionary
    -model_path: path to the model folder containing 1 folder per fold
    -output_folder: path to the output folder

Returns: 
    -output_image: path to the output image

Author: Pierre-Louis Benveniste     
"""
import argparse
import os
from pathlib import Path
import torch
from image import Image

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


def run_inference(input_image, model_path, fold_number, temp_folder):
    """
    This scripts runs the inference on a single image on only one fold at a time.
    """
    # Initialize the model 
    predictor = nnUNetPredictor(
        tile_step_size=0.5,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,                      # applies gaussian noise and gaussian blur
        use_mirroring=True,                    # test time augmentation by mirroring on all axes
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
        
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=[fold_number],
        checkpoint_name='checkpoint_best.pth',
    )

    # Run inference on the input image
    ## The inference worked when using python main.py -i... back when I was using predictor.predict_from_files
    ## However, when I ran with the Docker I had silent crashes or bus errors. Using predict_from_files_sequential allowed me to solve the problem of silent crash by after the inference.
    ## I found a temporary fix which was to set `--ipc=host` when running with the docker.
    ## However, this flag could not be used when running the inference with the Boutique
    ## Using `predict_from_files_sequential` seems to solve the problem of bus error and silent crash.: issue related: https://github.com/ivadomed/ms-multi-spine-challenge-2024/issues/40
    predictor.predict_from_files_sequential(
        list_of_lists_or_source_folder=[[input_image]],
        output_folder_or_list_of_truncated_output_files=temp_folder,
        save_probabilities=False,
        overwrite=True,
        folder_with_segs_from_prev_stage=None
    )
    
    # Build output image path (output folder and image name)
    output_image = os.path.join(temp_folder, Path(input_image).name.replace('_file', ''))

    # Rename the output image to include the fold number
    output_image_new = output_image.replace('.nii.gz', f'_fold{fold_number}.nii.gz')
    assert os.system(f"mv {output_image} {output_image_new}") == 0

    return output_image_new


def run_inference_on_all_images(subj_dict, model_path, output_folder):
    """
    This scripts runs the inference on all the images of the subject.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder in the output folder
    temp_folder = os.path.join(output_folder, "temp_inference")
    os.makedirs(temp_folder, exist_ok=True)

    # List of inference files
    inference_files = [subj_dict['t2_inference_file']]
    for other_image in subj_dict['other_images']:
        inference_files.append(other_image['inference_file'])

    # We iterate over the inference files and run inference on each of them
    for i, inference_file in enumerate(inference_files):
        ## Reorient the inference file to the RPI orientation
        reorient_inference_file = os.path.join(temp_folder, "reorient_inference_file.nii.gz")
        assert os.system(f"sct_image -i {inference_file} -setorient RPI -o {reorient_inference_file}") == 0
        ## Initialize the pred_path list
        file_preds = []
        ## Inference should be run on the 5 folds of the model
        for fold_nb in range(5):
            print(f"Running inference on the image (fold {fold_nb})...")
            # Build the path to the model for the fold
            model_path_fold = os.path.join(model_path, f"model_fold{fold_nb}")
            # Run inference
            pred_fold_i = run_inference(reorient_inference_file, model_path_fold, fold_nb, temp_folder)
            # Append the prediction to the list
            file_preds.append(pred_fold_i)
        ## Aggregate the predictions
        pred_aggregated = os.path.join(temp_folder, "pred_aggregated.nii.gz")
        assert os.system(f"sct_maths -i {file_preds[0]} -add {file_preds[1]} {file_preds[2]} {file_preds[3]} {file_preds[4]} -o {pred_aggregated} -type float64") == 0
        pred_avg = os.path.join(temp_folder, "pred_avg.nii.gz")
        assert os.system(f"sct_maths -i {pred_aggregated} -div 5 -o {pred_avg}") == 0
        ## Move the predictions back to the original orientation
        inference_file_orientation = Image(inference_file).orientation
        assert os.system(f"sct_image -i {pred_avg} -setorient {inference_file_orientation} -o {pred_avg}") == 0
        ## We register the prediction back to the T2w raw image
        pred_reg = os.path.join(temp_folder, "pred_reg.nii.gz")
        assert os.system(f"sct_register_multimodal -i {pred_avg} -d {subj_dict['t2_raw']} -identity 1 -o {pred_reg}") == 0
        ## Then we threshold the prediction at 0.1 to reduce the volume of the segmentation files
        assert os.system(f"sct_maths -i {pred_reg} -thr 0.1 -o {pred_reg}") == 0
        ## We copy the prediction to the output folder
        if i==0:
            output_segmentation = os.path.join(output_folder, "t2w_segmentation.nii.gz")
            subj_dict['t2_segmentation_file'] = output_segmentation
        else:
            output_segmentation = os.path.join(output_folder, f"{subj_dict['other_images'][i-1]['contrast']}_segmentation.nii.gz")
            subj_dict['other_images'][i-1]['segmentation_file'] = output_segmentation
        assert os.system(f"cp {pred_reg} {output_segmentation}") == 0
        
    # Clean up the temporary folder
    # assert os.system(f"rm -rf {temp_folder}") == 0

    return subj_dict

if __name__ == "__main__":
    args = parse_args()
    subj_dict = run_inference(args.input_image, args.output_folder)