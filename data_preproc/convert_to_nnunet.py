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
# import os
# import shutil
# import json
# import nibabel as nib
# import numpy as np
# import yaml
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


# def convert_to_nnUNet_format(root_dir, output_dir, path_data_split, task_name, kind, multimodal):
#     print(multimodal)
#     #Create the path to the nnUNet directory to put the dataset 
#     nnUNet_base = os.path.join(output_dir, "nnUNet_raw", task_name)
#     imagesTr = os.path.join(nnUNet_base, "imagesTr")
#     labelsTr = os.path.join(nnUNet_base, "labelsTr")
#     imagesTs = os.path.join(nnUNet_base, "imagesTs")
#     #If this is the first time running the script, create the directories
#     os.makedirs(imagesTr, exist_ok=True)
#     os.makedirs(labelsTr, exist_ok=True)
#     os.makedirs(imagesTs, exist_ok=True)


    
#     # Load the data split
#     with open(path_data_split, 'r') as file:
#         data_split_yml = yaml.load(file, Loader=yaml.FullLoader)

#     # Split into the training and testing
#     training_subjects = data_split_yml['TRAINING']
#     testing_subjects = data_split_yml['TESTING']



#     # List of contrasts and particles to generate the files names
#     contrast_list = ["STIR", "PSIR", "MP2RAGE"]
#     particle_list = {"base": "", "preproc": "desc-preproc_", "preproc_reg": "desc-preprocReg_"}

#     # Loop over the training subjects to convert the data to nnUNet format

#     for sub in training_subjects:
#         anat_dir = os.path.join(root_dir, sub, "anat")
#         label_dir = os.path.join(root_dir, "derivatives", "labels", sub, "anat")

#         t2w_image = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}T2w.nii.gz")
#         label_image = os.path.join(label_dir, f"{sub}_{particle_list[kind]}T2w_label-lesion_seg.nii.gz")

#         assert os.system(f"sct_image -i {t2w_image} -setorient RPI -o {imagesTr}/{sub}_0000.nii.gz") == 0

#         if multimodal: 
#             contrast_image = None
#             for contrast in contrast_list:
#                 contrast_path = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}{contrast}.nii.gz")
#                 if os.path.exists(contrast_path):
#                     contrast_image = contrast_path
#                     break

#             assert os.system(f"sct_image -i {contrast_image} -setorient RPI -o {imagesTr}/{sub}_0001.nii.gz") == 0


#         if os.path.exists(label_image):
#             label_nifti = nib.load(label_image)
#             label_data = label_nifti.get_fdata()

#             # Convert labels: set non-zero voxels to 1, keep zeros as 0
#             binary_label_data = (label_data > 0).astype(np.int8)

#             new_label_nifti = nib.Nifti1Image(binary_label_data, affine=label_nifti.affine, header=label_nifti.header)
#             label_path = os.path.join(labelsTr, f"{sub}.nii.gz")
#             nib.save(new_label_nifti, label_path)
#             assert os.system(f"sct_image -i {label_path} -setorient RPI -o {label_path}") == 0

        

        
#     for sub in testing_subjects:
#         anat_dir = os.path.join(root_dir, sub, "anat")

#         t2w_image = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}T2w.nii.gz")

#         assert os.system(f"sct_image -i {t2w_image} -setorient RPI -o {imagesTs}/{sub}_0000.nii.gz") == 0

#         if multimodal: 
#             contrast_image = None
#             for contrast in contrast_list:
#                 contrast_path = os.path.join(anat_dir, f"{sub}_{particle_list[kind]}{contrast}.nii.gz")
#                 if os.path.exists(contrast_path):
#                     contrast_image = contrast_path
#                     break

#             assert os.system(f"sct_image -i {contrast_image} -setorient RPI -o {imagesTs}/{sub}_0001.nii.gz") == 0


#     if multimodal:

#         dataset_json = {
#             "channel_names": {"0": "T2", "1": "Contrast"},
#             "labels": {"background": 0, "lesion": 1},
#             "numTraining": len(training_subjects),
#             "file_ending": ".nii.gz"
#         }

#     else:   
#         dataset_json = {
#             "channel_names": {"0": "T2"},
#             "labels": {"background": 0, "lesion": 1},
#             "numTraining": len(training_subjects),
#             "file_ending": ".nii.gz"
#         }

#     with open(os.path.join(nnUNet_base, "dataset.json"), "w") as f:
#         json.dump(dataset_json, f, indent=4)

#     print(f"Dataset successfully converted to nnUNet format at {nnUNet_base}")

def main():
    # Parse the arguments
    args = parse_arguments()

    # First we aggregated the data depending on the dataset type
    aggregated_data = agg_data(args.data, args.dataset_type)
    if args.dataset_type in [6, 7, 8]:
        input1_data, input2_data = aggregated_data
    
    # convert_to_nnUNet_format(args.root_dir, args.output_dir, args.path_data_split, args.task_name, args.kind_of_image, args.multimodal)

if __name__ == "__main__":
    main()
