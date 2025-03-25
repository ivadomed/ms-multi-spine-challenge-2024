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
# import shutil
# import json
import nibabel as nib
# import numpy as np
import yaml
import argparse
from pathlib import Path


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


def agg_data(data_dir, dataset_type):
    """
    This function aggregates the data depending on the dataset type.
    """
    if dataset_type == 1: # Data for dataset type 1 (monomodal_T2w_raw)
        # List all raw T2w images
        raw_t2w_images = Path(data_dir).rglob("*_T2w.nii.gz")
        # Remove images with "preproc" in the name
        raw_t2w_images = [str(p) for p in raw_t2w_images if "preproc" not in str(p)]
        return raw_t2w_images
    
    elif dataset_type == 2: # Data for dataset type 2 (monomodal_all_raw)
        # List all raw images
        raw_images = Path(data_dir).rglob("*.nii.gz")
        # Remove images with SHA256 in the name
        raw_images = [str(p) for p in raw_images if "SHA256" not in str(p)]
        # Remove images with "preproc" in the name
        raw_images = [str(p) for p in raw_images if "preproc" not in str(p)]
        # Remove derivative files
        raw_images = [str(p) for p in raw_images if "derivatives" not in str(p)]
        return raw_images
    
    elif dataset_type == 3: # Data for dataset type 3 (monomodal_T2w_preprocReg)
        # List all preprocReg T2w images
        preprocReg_t2w_images = Path(data_dir).rglob("*desc-preprocReg_T2w.nii.gz")
        preprocReg_t2w_images = [str(p) for p in preprocReg_t2w_images]
        return preprocReg_t2w_images
    
    elif dataset_type == 4: # Data for dataset type 4 (monomodal_all_preprocReg)
        # List all preprocReg images
        preprocReg_images = Path(data_dir).rglob("*desc-preprocReg*.nii.gz")
        # Remove derivative files
        preprocReg_images = [str(p) for p in preprocReg_images if "derivatives" not in str(p)]
        return preprocReg_images
    
    elif dataset_type == 5: # Data for dataset type 5 (monomodal_T2w_all)
        # List all T2w images
        t2w_images = Path(data_dir).rglob("*_T2w.nii.gz")
        # Remove preprocReg images
        t2w_images = [str(p) for p in t2w_images if "preprocReg" not in str(p)]
        return t2w_images
    
    elif dataset_type == 6 or dataset_type == 7: # Data for dataset type 6 (multimodal_raw_unregistered) and 7 (multimodal_all_unregistered)
        # List all T2w images for input 1
        input1_raw_T2w = Path(data_dir).rglob("*_T2w.nii.gz")
        input1_raw_T2w = [str(p) for p in input1_raw_T2w if "preproc" not in str(p)]
        # List all other contrasts of input 2
        input2_raw_contrasts = Path(data_dir).rglob("*.nii.gz")
        # Remove files with SHA256 in the name
        input2_raw_contrasts = [str(p) for p in input2_raw_contrasts if "SHA256" not in str(p)]
        # Remove T2w images
        input2_raw_contrasts = [str(p) for p in input2_raw_contrasts if "_T2w" not in str(p)]
        # Remove derivative files
        input2_raw_contrasts = [str(p) for p in input2_raw_contrasts if "derivatives" not in str(p)]
        # Remove preprocessed files
        input2_raw_contrasts = [str(p) for p in input2_raw_contrasts if "preproc" not in str(p)]
        return input1_raw_T2w, input2_raw_contrasts
    
    elif dataset_type == 8: # Data for dataset type 8 (multimodal_preprocReg_registered)
        # List all preprocReg T2w images for input 1
        input1_preprocReg_T2w = Path(data_dir).rglob("*desc-preprocReg_T2w.nii.gz")
        input1_preprocReg_T2w = [str(p) for p in input1_preprocReg_T2w]
        # List all other contrasts of input 2
        input2_preprocReg_contrasts = Path(data_dir).rglob("*desc-preprocReg*.nii.gz")
        # Remove T2w images
        input2_preprocReg_contrasts = [str(p) for p in input2_preprocReg_contrasts if "_T2w" not in str(p)]
        # Remove derivative files
        input2_preprocReg_contrasts = [str(p) for p in input2_preprocReg_contrasts if "derivatives" not in str(p)]
        return input1_preprocReg_T2w, input2_preprocReg_contrasts

    else:
        raise ValueError(f"Dataset type {dataset_type} is not recognized.")


def convert_to_nnUNet_format(agg_data, output_dir, path_data_split, task_name, task_number, dataset_type):
    """
    This function converts the aggregated data to the nnUNet format.
    """

    # Split the aggregated data in case of multimodal
    if dataset_type in [6, 7, 8]:
        input1_data, input2_data = agg_data

    # Define the output path
    path_out = Path(os.path.join(output_dir, f'Dataset{task_number}_{task_name}'))
    # Define paths for train and test folders 
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))
    # Load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    # Make the directories
    path_out.mkdir(parents=True, exist_ok=True)
    path_out_imagesTr.mkdir(parents=True, exist_ok=True)
    path_out_imagesTs.mkdir(parents=True, exist_ok=True)
    path_out_labelsTr.mkdir(parents=True, exist_ok=True)
    path_out_labelsTs.mkdir(parents=True, exist_ok=True)

    # Initialise the conversion dict
    conversion_dict = {}

    # Initialise the number of scans in train and in test folder
    scan_cnt_train, scan_cnt_test = 0, 0
    
    # Load the data split
    with open(path_data_split, 'r') as file:
        data_split_yml = yaml.load(file, Loader=yaml.FullLoader)
    # Split into the training and testing
    training_subjects = data_split_yml['TRAINING']
    testing_subjects = data_split_yml['TESTING']

    # Convert the data to nnUNet format of the monomodal datasets
    if dataset_type in [1, 2, 3, 4, 5]:
        for file in agg_data:
            # Get the subject name
            subject_name = file.split('/')[-1].split('_')[0]
            # Check if the subject is in the training or testing set
            if subject_name in training_subjects:
                # Get corresponding label
                label_file = file.replace('.nii.gz', '_label-lesion_seg.nii.gz')
                label_file = label_file.replace('ms-multi-spine-challenge-2024', 'ms-multi-spine-challenge-2024/derivatives/labels')
                if not os.path.exists(label_file):
                    raise ValueError(f"Derivative file not found: {label_file}")
                # Add to the training list
                train_images.append(file)
                train_labels.append(label_file)
                # Build output paths
                out_image = path_out_imagesTr / f'{subject_name}_0000.nii.gz'
                out_label = path_out_labelsTr / f'{subject_name}.nii.gz'
                # Binarize the label and save it
                label_data = nib.load(label_file).get_fdata()
                label_data[label_data > 0] = 1
                label_file = nib.Nifti1Image(label_data, nib.load(label_file).affine)
                nib.save(label_file, out_label)
                # Reorient both image and label to desired orientation and location
                assert os.system(f"sct_image -i {file} -setorient RPI -o {out_image}") == 0
                assert os.system(f"sct_image -i {out_label} -setorient RPI -o {out_label}") == 0
            elif subject_name in testing_subjects:
                # Get corresponding label
                label_file = file.replace('.nii.gz', '_label-lesion_seg.nii.gz')
                label_file = label_file.replace('ms-multi-spine-challenge-2024', 'ms-multi-spine-challenge-2024/derivatives/labels')
                if not os.path.exists(label_file):
                    raise ValueError(f"Derivative file not found: {label_file}")
                # Add to the training list
                test_images.append(file)
                test_labels.append(label_file)
                # Build output paths
                out_image = path_out_imagesTs / f'{subject_name}_0000.nii.gz'
                out_label = path_out_labelsTs / f'{subject_name}.nii.gz'
                # Binarize the label and save it
                label_data = nib.load(label_file).get_fdata()
                label_data[label_data > 0] = 1
                label_file = nib.Nifti1Image(label_data, nib.load(label_file).affine)
                nib.save(label_file, out_label)
                # Reorient both image and label to desired orientation and location
                assert os.system(f"sct_image -i {file} -setorient RPI -o {out_image}") == 0
                assert os.system(f"sct_image -i {out_label} -setorient RPI -o {out_label}") == 0


    # if multimodal:

    #     dataset_json = {
    #         "channel_names": {"0": "T2", "1": "Contrast"},
    #         "labels": {"background": 0, "lesion": 1},
    #         "numTraining": len(training_subjects),
    #         "file_ending": ".nii.gz"
    #     }

    # else:   
    #     dataset_json = {
    #         "channel_names": {"0": "T2"},
    #         "labels": {"background": 0, "lesion": 1},
    #         "numTraining": len(training_subjects),
    #         "file_ending": ".nii.gz"
    #     }

    # with open(os.path.join(nnUNet_base, "dataset.json"), "w") as f:
    #     json.dump(dataset_json, f, indent=4)

    # print(f"Dataset successfully converted to nnUNet format at {nnUNet_base}")

def main():
    # Parse the arguments
    args = parse_arguments()

    # First we aggregated the data depending on the dataset type
    aggregated_data = agg_data(args.data, args.dataset_type)

    # Then we convert it to the nnUnet format
    convert_to_nnUNet_format(aggregated_data, args.output, args.path_data_split, args.task_name, args.task_number, args.dataset_type)

if __name__ == "__main__":
    main()