"""
This script is used to convert the ms-multi-spine challenge dataset from the BIDS format to the nnUNet format.
The script will need to create the types of dataset detailed in the PR head: https://github.com/ivadomed/ms-multi-spine-challenge-2024/pull/13.
It takes as input a predefined data split (detailed in this issue: https://github.com/ivadomed/ms-multi-spine-challenge-2024/issues/12). 

Input: 
    --data: Path to the root directory of the dataset.
    --output: Path to the output directory where the nnUNet dataset will be created (nnUnet_raw).
    --path-data-split: Path to the data split YAML file.
    --task-name: Name of the task for nnUNet.
    --task-number: Number of the task for nnUnet (it has to be 3 digits).
    --dataset-type: Number corresponding to the dataset type we want to create. 

Output:
    None

Author: Thomas Dagonneau and Pierre-Louis Benveniste

TODO: 
    - 
"""
import os
import shutil
import json
import nibabel as nib
import numpy as np
import yaml
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory where the nnUNet dataset will be created (nnUnet_raw).")
    parser.add_argument("--path-data-split", type=str, required=True, help="Path to the data split YAML file.")
    parser.add_argument("--task-name", type=str, default="MsMultiSpine", help="Name of the task for nnUNet.")
    parser.add_argument("--task-number", type=str, required=True, help="Number of the task for nnUnet (it has to be 3 digits).")
    parser.add_argument("--dataset-type", type=int, required=True, help="Number corresponding to the dataset type we want to create.")
    args = parser.parse_args()
    return args


def convert_to_nnUNet_format(root_dir, output_dir, path_data_split, task_name, kind, multimodal):
    print(multimodal)
    #Create the path to the nnUNet directory to put the dataset 
    nnUNet_base = os.path.join(output_dir, "nnUNet_raw", task_name)
    imagesTr = os.path.join(nnUNet_base, "imagesTr")
    labelsTr = os.path.join(nnUNet_base, "labelsTr")
    imagesTs = os.path.join(nnUNet_base, "imagesTs")
    #If this is the first time running the script, create the directories
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)


    
    # Load the data split
    with open(path_data_split, 'r') as file:
        data_split_yml = yaml.load(file, Loader=yaml.FullLoader)

    # Split into the training and testing
    training_subjects = data_split_yml['TRAINING']
    testing_subjects = data_split_yml['TESTING']



    # List of contrasts and particles to generate the files names
    contrast_list = ["STIR", "PSIR", "MP2RAGE"]
    particle_list = {"base": "", "preproc": "desc-preproc_", "preproc_reg": "desc-preprocReg_"}

    # Loop over the training subjects to convert the data to nnUNet format

    for sub in training_subjects:
        anat_dir = os.path.join(root_dir, sub, "anat")
        label_dir = os.path.join(root_dir, "derivatives", "labels", sub, "anat")

        t2w_image = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}T2w.nii.gz")
        label_image = os.path.join(label_dir, f"{sub}_{particle_list[kind]}T2w_label-lesion_seg.nii.gz")

        assert os.system(f"sct_image -i {t2w_image} -setorient RPI -o {imagesTr}/{sub}_0000.nii.gz") == 0

        if multimodal: 
            contrast_image = None
            for contrast in contrast_list:
                contrast_path = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}{contrast}.nii.gz")
                if os.path.exists(contrast_path):
                    contrast_image = contrast_path
                    break

            assert os.system(f"sct_image -i {contrast_image} -setorient RPI -o {imagesTr}/{sub}_0001.nii.gz") == 0


        if os.path.exists(label_image):
            label_nifti = nib.load(label_image)
            label_data = label_nifti.get_fdata()

            # Convert labels: set non-zero voxels to 1, keep zeros as 0
            binary_label_data = (label_data > 0).astype(np.int8)

            new_label_nifti = nib.Nifti1Image(binary_label_data, affine=label_nifti.affine, header=label_nifti.header)
            label_path = os.path.join(labelsTr, f"{sub}.nii.gz")
            nib.save(new_label_nifti, label_path)
            assert os.system(f"sct_image -i {label_path} -setorient RPI -o {label_path}") == 0

        

        
    for sub in testing_subjects:
        anat_dir = os.path.join(root_dir, sub, "anat")

        t2w_image = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}T2w.nii.gz")

        assert os.system(f"sct_image -i {t2w_image} -setorient RPI -o {imagesTs}/{sub}_0000.nii.gz") == 0

        if multimodal: 
            contrast_image = None
            for contrast in contrast_list:
                contrast_path = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}{contrast}.nii.gz")
                if os.path.exists(contrast_path):
                    contrast_image = contrast_path
                    break

            assert os.system(f"sct_image -i {contrast_image} -setorient RPI -o {imagesTs}/{sub}_0001.nii.gz") == 0


    if multimodal:

        dataset_json = {
            "channel_names": {"0": "T2", "1": "Contrast"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(training_subjects),
            "file_ending": ".nii.gz"
        }

    else:   
        dataset_json = {
            "channel_names": {"0": "T2"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(training_subjects),
            "file_ending": ".nii.gz"
        }

    with open(os.path.join(nnUNet_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Dataset successfully converted to nnUNet format at {nnUNet_base}")

def main():
    # Parse the arguments
    args = parse_arguments()

    convert_to_nnUNet_format(args.root_dir, args.output_dir, args.path_data_split, args.task_name, args.kind_of_image, args.multimodal)

if __name__ == "__main__":
    main()
