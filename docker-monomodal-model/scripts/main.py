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
from run_inference import run_inference_on_all_images
from remove_lesions_outside_sc import remove_lesions_outside_sc
from remove_lesions_max_value import remove_lesions_max_value
from merge_predictions import merge_predictions
from binarize_prediction import binarize_prediction
from remove_small_lesions import remove_small_lesions
from format_segmentation import format_segmentation


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

    # subj_dict = {'t2_raw': '../sub-001/rawdata/sub-001/11-001_T2.nii.gz', 't2_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_T2.nii.gz', 't2_inference_file': '../output_sub-001/temp/t2w_inference.nii.gz', 'other_images': [{'image_raw': '../sub-001/rawdata/sub-001/11-001_STIR.nii.gz', 'image_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_STIR.nii.gz', 'contrast': 'STIR', 'inference_file': '../output_sub-001/temp/STIR_inference.nii.gz'}]}

    # 3. Now we perform inference on the preprocessed images
    subj_dict = run_inference_on_all_images(subj_dict, model_path, temp_folder)

    # subj_dict = {'t2_raw': '../sub-001/rawdata/sub-001/11-001_T2.nii.gz', 't2_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_T2.nii.gz', 't2_inference_file': '../output_sub-001/temp/t2w_inference.nii.gz', 'other_images': [{'image_raw': '../sub-001/rawdata/sub-001/11-001_STIR.nii.gz', 'image_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_STIR.nii.gz', 'contrast': 'STIR', 'inference_file': '../output_sub-001/temp/STIR_inference.nii.gz', 'segmentation_file': '../output_sub-001/temp/STIR_segmentation.nii.gz'}], 't2_segmentation_file': '../output_sub-001/temp/t2w_segmentation.nii.gz'}

    #########################
    #### POST-PROCESSING ####
    #########################
    # 4. Now we remove lesions outside of the spinal cord
    subj_dict = remove_lesions_outside_sc(subj_dict, temp_folder)

    # subj_dict = {'t2_raw': '../sub-001/rawdata/sub-001/11-001_T2.nii.gz', 't2_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_T2.nii.gz', 't2_inference_file': '../output_sub-001/temp/t2w_inference.nii.gz', 'other_images': [{'image_raw': '../sub-001/rawdata/sub-001/11-001_STIR.nii.gz', 'image_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_STIR.nii.gz', 'contrast': 'STIR', 'inference_file': '../output_sub-001/temp/STIR_inference.nii.gz', 'segmentation_file': '../output_sub-001/temp/STIR_segmentation.nii.gz', 'segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/STIR_segmentation_rmv_lesions_outside_sc.nii.gz'}], 't2_segmentation_file': '../output_sub-001/temp/t2w_segmentation.nii.gz', 't2_segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/t2w_segmentation_rmv_lesions_outside_sc.nii.gz'}

    # 5. Now we remove lesions where max voxel value is below 0.8
    subj_dict = remove_lesions_max_value(subj_dict, temp_folder)

    # subj_dict = {'t2_raw': '../sub-001/rawdata/sub-001/11-001_T2.nii.gz', 't2_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_T2.nii.gz', 't2_inference_file': '../output_sub-001/temp/t2w_inference.nii.gz', 'other_images': [{'image_raw': '../sub-001/rawdata/sub-001/11-001_STIR.nii.gz', 'image_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_STIR.nii.gz', 'contrast': 'STIR', 'inference_file': '../output_sub-001/temp/STIR_inference.nii.gz', 'segmentation_file': '../output_sub-001/temp/STIR_segmentation.nii.gz', 'segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/STIR_segmentation_rmv_lesions_outside_sc.nii.gz', 'segmentation_file_rmv_lesions_max_value': '../output_sub-001/temp/STIR_segmentation_file_rmv_lesions_max_value.nii.gz'}], 't2_segmentation_file': '../output_sub-001/temp/t2w_segmentation.nii.gz', 't2_segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/t2w_segmentation_rmv_lesions_outside_sc.nii.gz', 't2_segmentation_file_rmv_lesions_max_value': '../output_sub-001/temp/t2w_segmentation_file_rmv_lesions_max_value.nii.gz'}

    # 6. Merge predictions across contrasts
    subj_dict = merge_predictions(subj_dict, temp_folder)

    # 7. Binarization of the merged predictions (threshold = 0.8)
    subj_dict = binarize_prediction(subj_dict, temp_folder)

    # 8. Remove small lesions  (less than 18 voxels)
    subj_dict = remove_small_lesions(subj_dict, temp_folder)

    subj_dict = {'t2_raw': '../sub-001/rawdata/sub-001/11-001_T2.nii.gz', 't2_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_T2.nii.gz', 't2_inference_file': '../output_sub-001/temp/t2w_inference.nii.gz', 'other_images': [{'image_raw': '../sub-001/rawdata/sub-001/11-001_STIR.nii.gz', 'image_preproc': '../sub-001/derivatives/preprocessed/sub-001/11-001_STIR.nii.gz', 'contrast': 'STIR', 'inference_file': '../output_sub-001/temp/STIR_inference.nii.gz', 'segmentation_file': '../output_sub-001/temp/STIR_segmentation.nii.gz', 'segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/STIR_segmentation_rmv_lesions_outside_sc.nii.gz', 'segmentation_file_rmv_lesions_max_value': '../output_sub-001/temp/STIR_segmentation_file_rmv_lesions_max_value.nii.gz'}], 't2_segmentation_file': '../output_sub-001/temp/t2w_segmentation.nii.gz', 't2_segmentation_file_rmv_lesions_outside_sc': '../output_sub-001/temp/t2w_segmentation_rmv_lesions_outside_sc.nii.gz', 't2_segmentation_file_rmv_lesions_max_value': '../output_sub-001/temp/t2w_segmentation_file_rmv_lesions_max_value.nii.gz', 'merged_lesion_mask': '../output_sub-001/temp/merged_lesion_mask.nii.gz', 'binarized_lesion_mask': '../output_sub-001/temp/binarized_lesion_mask.nii.gz', 'lesion_mask_rmv_small_lesion': '../output_sub-001/temp/lesion_mask_rmv_small_lesion.nii.gz'}

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