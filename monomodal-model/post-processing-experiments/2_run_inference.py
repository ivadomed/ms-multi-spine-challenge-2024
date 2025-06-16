"""
In this script we run the inference (for model 151).

Author: Pierre-Louis Benveniste
"""
import json
import os
# We define the environment variables here to avoid a warning from nnunetv2
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results']="./nnUNet_results"
# Import for nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import torch
import argparse
from image import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for experiment 251.")
    parser.add_argument('--gpu', action='store_true', help='GPU to use for inference. Default is 0.')
    parser.add_argument("--image_dict", type=str, required=True, help="Path to the JSON file containing image metadata.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where subject folders will be created.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model folder.")
    return parser.parse_args()

def main():
    args = parse_args()
    image_dict = args.image_dict
    output_folder = args.output_folder
    model_path = args.model_path

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    # iterate over the images
    for image in images:
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        print("Subject name", images[image]["subject_name"])
        # We build the output folder
        output_folder_sub = os.path.join(sub_folder, "output")
        os.makedirs(output_folder_sub, exist_ok=True)
        # We build path to inference files
        t2_inference_file = os.path.join(sub_folder, "cropped_reg_t2w_preproc_to_t2w_raw.nii.gz")
        psir_inference_file = os.path.join(sub_folder, "cropped_reg_psir_preproc_to_t2_raw.nii.gz")

        # We build the temp folder
        temp_folder = os.path.join(sub_folder, "temp_pred")
        os.makedirs(temp_folder, exist_ok=True)

        # Build the paths
        reorient_t2_inference_file = os.path.join(temp_folder, "reorient_t2w_inference_file.nii.gz")
        reorient_psir_inference_file = os.path.join(temp_folder, "reorient_psir_inference_file.nii.gz")

        # We reorient both image to the model orientation
        assert os.system(f"sct_image -i {t2_inference_file} -setorient RPI -o {reorient_t2_inference_file}") == 0
        assert os.system(f"sct_image -i {psir_inference_file} -setorient RPI -o {reorient_psir_inference_file}") == 0    

        # Now we run inference with the 5 folds on both T2 and PSIR images
        for fold in range(5):
            print(f"\nRunning inference for fold {fold} on subject {images[image]['subject_name']}")
            # Initialize the model 
            predictor = nnUNetPredictor(
                tile_step_size=0.5,     # changing it from 0.5 to 0.9 makes inference faster
                use_gaussian=True,                      # applies gaussian noise and gaussian blur
                use_mirroring=True,                    # test time augmentation by mirroring on all axes
                device=torch.device('cuda') if args.gpu else torch.device('cpu'),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
                
            # initializes the network architecture, loads the checkpoint
            predictor.initialize_from_trained_model_folder(
                model_path,
                use_folds=[fold],
                checkpoint_name='checkpoint_best.pth',
            )

            # Run inference on the T2w image
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[reorient_t2_inference_file]],
                output_folder_or_list_of_truncated_output_files=temp_folder,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=8,
                num_processes_segmentation_export=8,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )

            # Run inference on the PSIR image
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[reorient_psir_inference_file]],
                output_folder_or_list_of_truncated_output_files=temp_folder,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=8,
                num_processes_segmentation_export=8,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
            # We rename the output files: 
            output_t2_pred = os.path.join(output_folder_sub, images[image]['t2w_raw_image'].split("/")[-1].replace('.nii.gz',f'_fold{fold}.nii.gz'))
            output_psir_pred = os.path.join(output_folder_sub,  images[image]['t2w_raw_image'].split("/")[-1].replace("T2w", images[image]['contrast']).replace('.nii.gz',f'_fold{fold}.nii.gz'))
            
            assert os.system(f"mv {reorient_t2_inference_file.replace('_file','')} {output_t2_pred}") == 0
            assert os.system(f"mv {reorient_psir_inference_file.replace('_file','')} {output_psir_pred}") == 0
        
        # Remove the temp folder
        assert os.system(f"rm -rf {temp_folder}") == 0
        
        # Now we sum and average the predictions
        ## Add the T2w predictions
        assert os.system(f"sct_maths -i {output_t2_pred.replace('fold4','fold0')} -add {output_t2_pred.replace('fold4','fold1')} {output_t2_pred.replace('fold4','fold2')} {output_t2_pred.replace('fold4','fold3')} {output_t2_pred} -o {output_t2_pred.replace('fold4','sum')} -type float64") == 0
        assert os.system(f"sct_maths -i {output_t2_pred.replace('fold4','sum')} -div 5 -o {output_t2_pred.replace('fold4','avg')}") == 0

        ## Add the PSIR predictions
        assert os.system(f"sct_maths -i {output_psir_pred.replace('fold4','fold0')} -add {output_psir_pred.replace('fold4','fold1')} {output_psir_pred.replace('fold4','fold2')} {output_psir_pred.replace('fold4','fold3')} {output_psir_pred} -o {output_psir_pred.replace('fold4','sum')} -type float64") == 0
        assert os.system(f"sct_maths -i {output_psir_pred.replace('fold4','sum')} -div 5 -o {output_psir_pred.replace('fold4','avg')}") == 0

        # Move the predictions back to the original orientation
        t2_inference_file_orientation = Image(t2_inference_file).orientation
        psir_inference_file_orientation = Image(psir_inference_file).orientation
        assert os.system(f"sct_image -i {output_t2_pred.replace('fold4','avg')} -setorient {t2_inference_file_orientation} -o {output_t2_pred.replace('fold4','avg_reoriented')}") == 0
        assert os.system(f"sct_image -i {output_psir_pred.replace('fold4','avg')} -setorient {psir_inference_file_orientation} -o {output_psir_pred.replace('fold4','avg_reoriented')}") == 0

        # We bring both images back to the T2w raw space
        ## For the PSIR pred
        assert os.system(f"sct_register_multimodal -i {output_psir_pred.replace('fold4','avg_reoriented')} -d {images[image]['t2w_raw_image']} -identity 1 -o {output_psir_pred.replace('fold4','avg_reoriented_final')}") == 0
        ## For the T2w pred
        assert os.system(f"sct_register_multimodal -i {output_t2_pred.replace('fold4','avg_reoriented')} -d {images[image]['t2w_raw_image']} -identity 1 -o {output_t2_pred.replace('fold4','avg_reoriented_final')}") == 0

        # Then we threshold both predictions at 0.1 to reduce the volume of the segmentation files
        assert os.system(f"sct_maths -i {output_t2_pred.replace('fold4','avg_reoriented_final')} -thr 0.1 -o {output_t2_pred.replace('fold4','avg_reoriented_final')}") == 0
        assert os.system(f"sct_maths -i {output_psir_pred.replace('fold4','avg_reoriented_final')} -thr 0.1 -o {output_psir_pred.replace('fold4','avg_reoriented_final')}") == 0

        # We only keep the final files which we copy in a pred_folder
        pred_folder = os.path.join(sub_folder, "predictions")
        os.makedirs(pred_folder, exist_ok=True)
        assert os.system(f"cp {output_t2_pred.replace('fold4','avg_reoriented_final')} {pred_folder}/t2w_segmentation.nii.gz") == 0
        assert os.system(f"cp {output_psir_pred.replace('fold4','avg_reoriented_final')} {pred_folder}/psir_segmentation.nii.gz") == 0

        # Remove the output folder
        assert os.system(f"rm -rf {output_folder_sub}") == 0


if __name__ == '__main__':
    main()