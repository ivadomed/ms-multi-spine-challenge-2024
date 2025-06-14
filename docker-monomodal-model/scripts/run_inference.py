"""
This script runs the inference on the images using the 5 folds of the model.

Input: 
    -subj-dict: path to the subject dictionary
    -output_folder: path to the output folder

Returns: 
    -output_image: path to the output image

Author: Pierre-Louis Benveniste     
"""
import argparse
import os
from pathlib import Path
from image import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def run_inference(input_folder, fold_number, temp_folder):
    """
    This scripts runs the inference on a single image on only one fold at a time.
    """
    # We create a temp folder for the fold
    temp_folder_fold = os.path.join(temp_folder, f"fold_{fold_number}")
    os.makedirs(temp_folder_fold, exist_ok=True)

    # Run inference
    print(f"nnUNetv2_predict -i {input_folder} -o {temp_folder_fold} -d 151 -p nnUNetResEncUNetLPlansFinetune -tr nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch -c 3d_fullres -f {fold_number} -chk checkpoint_best.pth -device cpu")
    assert os.system(f"nnUNetv2_predict -i {input_folder} -o {temp_folder_fold} -d 151 -p nnUNetResEncUNetLPlansFinetune -tr nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs_fromScratch -c 3d_fullres -f {fold_number} -chk checkpoint_best.pth -device cpu") == 0

    # output image is the only file in the temp folder_fold
    output_image = list(Path(temp_folder_fold).rglob("*.nii.gz"))[0]

    # Rename the output image to include the fold number
    output_image_new = str(output_image).replace('.nii.gz', f'_fold{fold_number}.nii.gz')
    assert os.system(f"mv {output_image} {output_image_new}") == 0

    return output_image_new


def run_inference_on_all_images(subj_dict, output_folder):
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

        ## Move inference file to a temp folder
        temp_inference_nnunet_folder = os.path.join(temp_folder, "nnunet_inference")
        os.makedirs(temp_inference_nnunet_folder, exist_ok=True)
        assert os.system(f"cp {reorient_inference_file} {temp_inference_nnunet_folder}/file_0000.nii.gz") == 0

        # print the content of temp_inference_nnunet_folder
        print("Content of temp_inference_nnunet_folder:")
        print(os.listdir(temp_inference_nnunet_folder))

        ## Inference should be run on the 5 folds of the model
        for fold_nb in range(5):
            print(f"Running inference on the image (fold {fold_nb})...")
            # Run inference
            pred_fold_i = run_inference(temp_inference_nnunet_folder, fold_nb, temp_folder)
            # Append the prediction to the list
            file_preds.append(pred_fold_i)

        # Remove the temp_inference_nnunet_folder
        # assert os.system(f"rm -rf {temp_inference_nnunet_folder}") == 0

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

# if __name__ == "__main__":
#     args = parse_args()
#     subj_dict = run_inference(args.input_image, args.output_folder)