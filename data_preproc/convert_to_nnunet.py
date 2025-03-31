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
import json
import nibabel as nib
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np 


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
    
    elif dataset_type == 4 or dataset_type == 10: # Data for dataset type 4 (monomodal_all_preprocReg) and 10 (monomodal_all_preprocReg_croppped)
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
    
    elif dataset_type == 8 or dataset_type == 9 : # Data for dataset type 8 (multimodal_preprocReg_registered) and 9 (multimodal_preprocReg_registered_cropped)
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


# Function to extract the coordinates to crop from  
def get_nonzero_bbox(image_data):
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return max_idx, min_idx


def convert_to_nnUNet_format(agg_data, output_dir, path_data_split, task_name, task_number, dataset_type):
    """
    This function converts the aggregated data to the nnUNet format.
    """

    # Split the aggregated data in case of multimodal
    if dataset_type in [6, 7, 8, 9]:
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

    # Initialise the conversion dict and image_label_conversion_dict
    conversion_dict = {}
    image_label_conversion_dict = {}

    # Initialise the number of scans in train and in test folder
    scan_cnt_train, scan_cnt_test = 0, 0
    
    # Load the data split
    with open(path_data_split, 'r') as file:
        data_split_yml = yaml.load(file, Loader=yaml.FullLoader)
    # Split into the training and testing
    training_subjects = data_split_yml['TRAINING']
    testing_subjects = data_split_yml['TESTING']

    # Convert the data to nnUNet format of the monomodal datasets
    if dataset_type in [1, 2, 3, 4, 5, 10]:
        for file in agg_data:
            # Get the subject name
            subject_name = file.split('/')[-1].split('_')[0]
            contrast = file.split('/')[-1].split('_')[-1].split('.')[0]
            # Get corresponding label
            label_file = file.replace(f'{contrast}.nii.gz', 'T2w_label-lesion_seg.nii.gz')
            label_file = label_file.replace('ms-multi-spine-challenge-2024', 'ms-multi-spine-challenge-2024/derivatives/labels')
            if not os.path.exists(label_file):
                raise ValueError(f"Derivative file not found: {label_file}")
           
            # Check if the subject is in the training or testing set
            if subject_name in training_subjects:
                scan_cnt_train += 1
                # Add to the training list
                train_images.append(file)
                train_labels.append(label_file)
                # Build output paths
                out_image = path_out_imagesTr / f'{task_name}_{scan_cnt_train:03d}_0000.nii.gz'
                out_label = path_out_labelsTr / f'{task_name}_{scan_cnt_train:03d}.nii.gz'
                
            
            # For the testing files
            elif subject_name in testing_subjects:
                scan_cnt_test += 1
                # Add to the testing list
                test_images.append(file)
                test_labels.append(label_file)
                # Build output paths
                out_image = path_out_imagesTs / f'{task_name}_{scan_cnt_test:03d}_0000.nii.gz'
                out_label = path_out_labelsTs / f'{task_name}_{scan_cnt_test:03d}.nii.gz'
            
            # Add to the conversion dict
            conversion_dict[str(file)] = str(out_image)
            conversion_dict[str(label_file)] = str(out_label)
            # Add to the image_label_conversion_dict
            sub_dict = {"image_source": str(file), "label_source": str(label_file), "image_nnunet": str(out_image), "label_nnunet": str(out_label)}
            image_label_conversion_dict[str(file)] = sub_dict
        
            
            # Reorient both image and label to desired orientation and location
            assert os.system(f"sct_image -i {file} -setorient RPI -o {out_image}") == 0
            assert os.system(f"sct_image -i {label_file} -setorient RPI -o {out_label}") == 0

            # An extra security is to use the sct_register_multimodal to match dimension, resolution and orientation
            assert os.system(f"sct_register_multimodal -i {str(out_label)} -d {str(out_image)} -identity 1 -o {str(out_label)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") == 0
            # Remove the other useless files
            assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") == 0
            other_file_to_remove = str(out_label).replace('.nii.gz', '_inv.nii.gz')
            assert os.system(f"rm {other_file_to_remove}") == 0


            # Binarize the label and save it
            label_data = nib.load(out_label).get_fdata()
            label_data[label_data > 0] = 1
            label_nifty = nib.Nifti1Image(label_data, nib.load(out_label).affine)
            nib.save(label_nifty, out_label)
    
           

            # For the dataset type 10, we crop the image around non-empty areas of the image
            if dataset_type == 10:
                # Load the image and get the coordinates
                image_data = nib.load(out_image).get_fdata()
                # Get the cropping box coordinates
                max_idx, min_idx = get_nonzero_bbox(image_data)
                # Crop the images using SCT
                assert os.system(f'sct_crop_image -i {out_image} -o {out_image} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
                assert os.system(f'sct_crop_image -i {out_label} -o {out_label} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 

    # In the multimodal case
    elif dataset_type in [6, 7, 8, 9]:
        for file1, file2 in zip(input1_data, input2_data):
            # Get the subject name
            subject_name = file1.split('/')[-1].split('_')[0]
            contrast = file1.split('/')[-1].split('_')[-1].split('.')[0]
            # Get corresponding label
            label_file = file1.replace(f'{contrast}.nii.gz', 'T2w_label-lesion_seg.nii.gz')
            label_file = label_file.replace('ms-multi-spine-challenge-2024', 'ms-multi-spine-challenge-2024/derivatives/labels')
            if not os.path.exists(label_file):
                raise ValueError(f"Derivative file not found: {label_file}")
            
            # Check if the subject is in the training or testing set
            if subject_name in training_subjects:
                scan_cnt_train += 1
                # Add to the training list
                train_images.append(file1)
                train_images.append(file2)
                train_labels.append(label_file)
                # Build output paths
                out_image1 = path_out_imagesTr / f'{task_name}_{scan_cnt_train:03d}_0000.nii.gz'
                out_image2 = path_out_imagesTr / f'{task_name}_{scan_cnt_train:03d}_0001.nii.gz'
                out_label = path_out_labelsTr / f'{task_name}_{scan_cnt_train:03d}.nii.gz'
            # For the testing files
            elif subject_name in testing_subjects:
                scan_cnt_test += 1
                # Add to the testing list
                test_images.append(file1)
                test_images.append(file2)
                test_labels.append(label_file)
                # Build output paths
                out_image1 = path_out_imagesTs / f'{task_name}_{scan_cnt_test:03d}_0000.nii.gz'
                out_image2 = path_out_imagesTs / f'{task_name}_{scan_cnt_test:03d}_0001.nii.gz'
                out_label = path_out_labelsTs / f'{task_name}_{scan_cnt_test:03d}.nii.gz'

            # Add to the conversion dict
            conversion_dict[str(file1)] = str(out_image1)
            conversion_dict[str(file2)] = str(out_image2)
            conversion_dict[str(label_file)] = str(out_label)
            # Add to the image_label_conversion_dict
            sub_dict = {"image1_source": str(file1), "image2_source": str(file2), "label_source": str(label_file), "image1_nnunet": str(out_image1), "image2_nnunet": str(out_image2), "label_nnunet": str(out_label)}
            image_label_conversion_dict[str(file1)] = sub_dict

            
            # Reorient both image and label to desired orientation and location
            assert os.system(f"sct_image -i {file1} -setorient RPI -o {out_image1}") == 0
            assert os.system(f"sct_image -i {file2} -setorient RPI -o {out_image2}") == 0
            assert os.system(f"sct_image -i {label_file} -setorient RPI -o {out_label}") == 0
            
            # An extra security is to use the sct_register_multimodal to match dimension, resolution and orientation
            assert os.system(f"sct_register_multimodal -i {str(out_label)} -d {str(out_image1)} -identity 1 -o {str(out_label)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") == 0
            # Remove the other useless files
            assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") == 0
            other_file_to_remove = str(out_label).replace('.nii.gz', '_inv.nii.gz')
            assert os.system(f"rm {other_file_to_remove}") == 0

            # Binarize the label and save it
            label_data = nib.load(out_label).get_fdata()
            label_data[label_data > 0] = 1
            label_nifty = nib.Nifti1Image(label_data, nib.load(out_label).affine)
            nib.save(label_nifty, out_label)

            # If dataset_type is 9, we need to crop the images
            ## The cropping is zone by finding the empty region on the T2w image
            ## That cropping is applied to image1, image2 and label
            if dataset_type == 9:
                # Load the image and get the coordinates
                image_data = nib.load(out_image1).get_fdata()
                # Get the cropping box coordinates
                max_idx, min_idx = get_nonzero_bbox(image_data)
                # Crop the images using SCT
                assert os.system(f'sct_crop_image -i {out_image1} -o {out_image1} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
                assert os.system(f'sct_crop_image -i {out_image2} -o {out_image2} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 
                assert os.system(f'sct_crop_image -i {out_label} -o {out_label} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 

    # Print number of images in training and testing
    print("Number of images for training: " + str(scan_cnt_train))
    print("Number of images for testing: " + str(scan_cnt_test))

    #----------------- CREATION OF THE DICTIONNARY-----------------------------------
    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)

    # c.f. dataset json generation. This contains the metadata for the dataset that nnUNet uses during preprocessing and training
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py
    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "MS Multi-Spine Challenge 2024"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    
    # Because only using one modality  
    ## was changed from 'modality' to 'channel_names'
    json_dict['channel_names'] = {
        "0": "MRI",
    }
    if dataset_type in [6, 7, 8]:
        json_dict['modality'] = {
            "0": "MRI",
            "1": "MRI"
        }
    
    # 0 is always the background. Any class labels should start from 1.
    json_dict['labels'] = {
        "background" : 0,
        "lesion" : 1,
    }
   
    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test
    #Newly required field in the json file with v2
    json_dict["file_ending"] = ".nii.gz"

    if dataset_type in [1, 2, 3, 4, 5]:
        json_dict['training'] = [{'image': str(train_images[i]) , "label": train_labels[i] }
                                    for i in range(len(train_labels))]
        json_dict['test'] = [{'image': str(test_images[i]) , "label": test_labels[i] }
                                    for i in range(len(test_images))]
    
    elif dataset_type in [6, 7, 8]:
        json_dict['training'] = [{'image1': str(train_images[2*i]), 'image2': str(train_images[2*i+1]), "label": train_labels[i] }
                                    for i in range(len(train_labels))]
        json_dict['test'] = [{'image1': str(test_images[2*i]), 'image2': str(test_images[2*i+1]), "label": test_labels[i] }
                                    for i in range(len(test_labels))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)
    print(f"Conversion done. The dataset is saved in {path_out}")

    return image_label_conversion_dict


def register_label_from_T2w(image_label_dict, output_dir):
    """
    This script is used to register the label to the T2w image.
    To do so, we need to create the warping field from T2w to the other contrasts and apply it the label.
    """
    # We iterate over the training dict
    for element in image_label_dict:
        image_source = image_label_dict[element]["image_source"]
        image_nnunet = image_label_dict[element]["image_nnunet"]
        label_nnunet = image_label_dict[element]["label_nnunet"]
        # if the image is not a T2w we need to modify the label
        if "T2w" not in image_source:
            # Get corresponding T2w image
            contrast = image_source.split('/')[-1].split('_')[-1].split('.')[0]
            T2w_image = image_source.replace(contrast, "T2w")
            # Build temporary folder in output path
            temp_folder = Path(output_dir) / "tmp"
            temp_folder.mkdir(parents=True, exist_ok=True)
            # Build warping field from T2w_image to image
            assert os.system(f"sct_register_multimodal -i {T2w_image} -d {image_source} -ofolder {temp_folder} -owarp {temp_folder/'warp_t2_to_stir.nii.gz'}") == 0
            # Apply the warping field to the label
            assert os.system(f"sct_apply_transfo -i {label_nnunet} -d {image_nnunet} -w {temp_folder/'warp_t2_to_stir.nii.gz'} -o {label_nnunet} -x nn") == 0
            # Binarize the label and save it
            label_data = nib.load(label_nnunet).get_fdata()
            label_data[label_data > 0.1] = 1
            label_nifti = nib.Nifti1Image(label_data, nib.load(label_nnunet).affine)
            nib.save(label_nifti, label_nnunet)
            # Remove the temporary folder
            assert os.system(f"rm -r {temp_folder}") == 0
    print("Label registration done.")


def register_2ndInput_to_T2wInput(image_label_dict, output_dir):
    """
    This function is used in the context of dataset_type 7.
    It registers the 2nd input (which is either STIR, PSIR or MP2RAGE) to the T2w input.
    """
    # We iterate over the images
    for element in image_label_dict:
        image1_nnunet = image_label_dict[element]["image1_nnunet"]
        image2_nnunet = image_label_dict[element]["image2_nnunet"]
        # Build temporary folder in output path
        temp_folder = Path(output_dir) / "tmp"
        temp_folder.mkdir(parents=True, exist_ok=True)
        # Register the 2nd input to the T2w input
        assert os.system(f"sct_register_multimodal -i {image2_nnunet} -d {image1_nnunet} -ofolder {temp_folder} -owarp {temp_folder/'warp_stir_to_t2.nii.gz'} -o {temp_folder/'input2_registered_to_input1.nii.gz'}") == 0
        # Copy the registered file to the nnunet folder
        assert os.system(f"cp {temp_folder/'input2_registered_to_input1.nii.gz'} {image2_nnunet}") == 0
        # Remove the temporary folder
        assert os.system(f"rm -r {temp_folder}") == 0
    print("2nd input registration done.")


def main():
    # Parse the arguments
    args = parse_arguments()

    # First we aggregated the data depending on the dataset type
    aggregated_data = agg_data(args.data, args.dataset_type)

    # Then we convert it to the nnUnet format
    image_label_dict = convert_to_nnUNet_format(aggregated_data, args.output, args.path_data_split, args.task_name, args.task_number, args.dataset_type)

    # In the case of dataset type 2, we need to register the label to the image.
    ## This means that we need to create the warping field from T2w to the other contrasts and apply it the label
    if args.dataset_type == 2:
        # Register the label to the T2w image
        register_label_from_T2w(image_label_dict, args.output)

    # In the case of dataset type 7, we need to register the other contrast to the T2w image.
    if args.dataset_type == 7:
        # Register the label to the T2w image
        register_2ndInput_to_T2wInput(image_label_dict, args.output)
    
    print("Conversion done.")


if __name__ == "__main__":
    main()
