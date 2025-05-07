"""
This script is used to generated nnUNet format dataset from the original dataset.
More precisely, it creates the dataset 11, 12, 13, 14 and 15: more details here: https://github.com/ivadomed/ms-multi-spine-challenge-2024/pull/13#issuecomment-2805024837

Input: 
    --data: Path to the root directory of the dataset.
    --output: Path to the output directory where the nnUNet dataset will be created (nnUnet_raw).
    --path-data-split: Path to the data split YAML file.
    --task-name: Name of the task for nnUNet.
    --task-number: Number of the task for nnUnet (it has to be 3 digits).
    --dataset-type: Number corresponding to the dataset type we want to create. 

Output:
    None

Author: Pierre-Louis Benveniste
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
    images = []
    if dataset_type == 15: # dataset 15 (monomodal_all_preprocReg_T2wspace)
        # List all preproc
        images = Path(data_dir).rglob("*desc-preproc*.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]

    elif dataset_type == 11: # dataset 11 (monomodal_T2w_preproc_T2wRawSpace)
        # List all preproc T2w images
        images = Path(data_dir).rglob("*desc-preproc*T2w.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]

    elif dataset_type == 12: # dataset 12 (monomodal_PSIR_preproc_T2wspace)
        # List all preproc PSIR images
        images = Path(data_dir).rglob("*desc-preproc*PSIR.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]

    elif dataset_type == 13: # dataset 13 (monomodal_STIR_preproc_T2wspace)
        # List all preproc STIR images
        images = Path(data_dir).rglob("*desc-preproc*STIR.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]

    elif dataset_type == 14: # dataset 14 (monomodal_MP2RAGE_preproc_T2wspace)
        # List all preproc MP2RAGE images
        images = Path(data_dir).rglob("*desc-preproc*MP2RAGE.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]

    elif dataset_type == 16: #dataset 16 (multimodal_all_preprocReg_T2wspace)
        # List all preproc
        images = Path(data_dir).rglob("*desc-preproc*.nii.gz")
        # Remove all images with desc-preprocReg in name
        images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
        # Remove all images with derivatives in name
        images = [str(img) for img in images if "derivatives" not in str(img)]
        #Select the images with T2w in the name
        images1 = [img for img in images if "T2w" in str(img)]
        # Select the images with T2w not in the name
        images2 = [img for img in images if "T2w" not in str(img)]
        #Sort the images
        images1.sort()
        images2.sort()
        return images1, images2 

    return images


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
    # Define the output path
    path_out = Path(os.path.join(output_dir, f'Dataset{task_number}_{task_name}'))
    # Define paths for train and test folders 
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))
    path_registration_qc = Path(os.path.join(path_out, 'registration_qc'))
    # Load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    # Make the directories
    path_out.mkdir(parents=True, exist_ok=True)
    path_out_imagesTr.mkdir(parents=True, exist_ok=True)
    path_out_imagesTs.mkdir(parents=True, exist_ok=True)
    path_out_labelsTr.mkdir(parents=True, exist_ok=True)
    path_out_labelsTs.mkdir(parents=True, exist_ok=True)
    path_registration_qc.mkdir(parents=True, exist_ok=True)

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

    if dataset_type==16:
        input1_data, input2_data = agg_data
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

            temp_folder = Path(output_dir) / "tmp"
            temp_folder.mkdir(parents=True, exist_ok=True)

            # We generate the QC:
            # Copy the T2w raw image to the temp folder
            t2w_raw_image = file1 
            assert os.system(f"cp {t2w_raw_image} {temp_folder/'raw_t2w.nii.gz'}") == 0
            ## First we need to generate the spinal cord segmentation
            assert os.system(f"sct_deepseg -i {temp_folder/'raw_t2w.nii.gz'} -o {temp_folder/'raw_t2w_sc_seg.nii.gz'} -task seg_sc_contrast_agnostic ") == 0

            # We register the image to the corresponding T2w raw space
            
            # If the image is a T2w image then we juste have to move it back to its original space
            assert os.system(f"sct_register_multimodal -i {file1} -d {t2w_raw_image} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw1.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw1.nii.gz'} -qc {path_registration_qc} -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} -qc-subject {file1.split('/')[-1]}") == 0
     
            # Else we need to compute more complex registration
            parameters = 'step=1,type=im,algo=dl'
            assert os.system(f"sct_register_multimodal -i {file2} -d {t2w_raw_image} -param {parameters} -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw2.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw2.nii.gz'} -qc {path_registration_qc} -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} -qc-subject {file2.split('/')[-1]}") == 0
    
            # Copy the registered file to the nnunet folder
            assert os.system(f"cp {temp_folder/'image_reg_to_t2wraw1.nii.gz'} {out_image1}") == 0
            assert os.system(f"cp {temp_folder/'image_reg_to_t2wraw2.nii.gz'} {out_image2}") == 0

            
            # Reorient both image and label to desired orientation and location
            assert os.system(f"sct_image -i {out_image1} -setorient RPI -o {out_image1}") == 0
            assert os.system(f"sct_image -i {out_image2} -setorient RPI -o {out_image2}") == 0
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

            
            # Load the image and get the coordinates
            image_data = nib.load(out_image1).get_fdata()
            # Get the cropping box coordinates
            max_idx, min_idx = get_nonzero_bbox(image_data)
            # Crop the images using SCT
            assert os.system(f'sct_crop_image -i {out_image1} -o {out_image1} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
            assert os.system(f'sct_crop_image -i {out_image2} -o {out_image2} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 
            assert os.system(f'sct_crop_image -i {out_label} -o {out_label} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 

    
    else: 

        for file in agg_data:
            # Get the subject name
            subject_name = file.split('/')[-1].split('_')[0]
            contrast = file.split('/')[-1].split('_')[-1].split('.')[0]
            # Get corresponding label
            label_file = file.replace(f'{contrast}.nii.gz', 'T2w_label-lesion_seg.nii.gz')
            label_file = label_file.replace('ms-multi-spine-challenge-2024', 'ms-multi-spine-challenge-2024/derivatives/labels')
            # Label file in raw T2w space
            label_file = label_file.replace('_desc-preproc', '')
            if not os.path.exists(label_file):
                raise ValueError(f"Derivative file not found: {label_file}")
            # Then we find the corresponding T2w raw image
            t2w_raw_image = file.replace(f'{contrast}.nii.gz', 'T2w.nii.gz')
            t2w_raw_image = t2w_raw_image.replace('_desc-preproc', '')
            if not os.path.exists(t2w_raw_image):
                raise ValueError(f"Derivative file not found: {t2w_raw_image}")
                            
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

            temp_folder = Path(output_dir) / "tmp"
            temp_folder.mkdir(parents=True, exist_ok=True)

            # We generate the QC:
            # Copy the T2w raw image to the temp folder
            assert os.system(f"cp {t2w_raw_image} {temp_folder/'raw_t2w.nii.gz'}") == 0
            ## First we need to generate the spinal cord segmentation
            assert os.system(f"sct_deepseg -i {temp_folder/'raw_t2w.nii.gz'} -o {temp_folder/'raw_t2w_sc_seg.nii.gz'} -task seg_sc_contrast_agnostic ") == 0

            # We register the image to the corresponding T2w raw space
            if contrast == 'T2w':
                # If the image is a T2w image then we juste have to move it back to its original space
                assert os.system(f"sct_register_multimodal -i {file} -d {t2w_raw_image} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw.nii.gz'} -qc {path_registration_qc} -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} -qc-subject {file.split('/')[-1]}") == 0
            else:
                # Else we need to compute more complex registration
                parameters = 'step=1,type=im,algo=dl'
                assert os.system(f"sct_register_multimodal -i {file} -d {t2w_raw_image} -param {parameters} -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw.nii.gz'} -qc {path_registration_qc} -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} -qc-subject {file.split('/')[-1]}") == 0
        
            # Copy the registered file to the nnunet folder
            assert os.system(f"cp {temp_folder/'image_reg_to_t2wraw.nii.gz'} {out_image}") == 0

            # Reorient both image and label to desired orientation and location
            assert os.system(f"sct_image -i {out_image} -setorient RPI -o {out_image}") == 0
            assert os.system(f"sct_image -i {label_file} -setorient RPI -o {out_label}") == 0
            
            # Remove the temporary folder
            assert os.system(f"rm -r {temp_folder}") == 0

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

            # We end by cropping the zero areas of the images
            # Load the image and get the coordinates
            image_data = nib.load(out_image).get_fdata()
            # Get the cropping box coordinates
            max_idx, min_idx = get_nonzero_bbox(image_data)
            # Crop the images using SCT
            assert os.system(f'sct_crop_image -i {out_image} -o {out_image} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0
            assert os.system(f'sct_crop_image -i {out_label} -o {out_label} -xmin {min_idx[0]} -ymin {min_idx[1]} -zmin {min_idx[2]} -xmax {max_idx[0]} -ymax {max_idx[1]} -zmax {max_idx[2]}') == 0 
    
    #----------------- CREATION OF THE CONVERSION DICT-----------------------------------
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
    if dataset_type == 16:
        json_dict['channel_names'] = {
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

def main():
    # Parse the arguments
    args = parse_arguments()

    # First we aggregated the data depending on the dataset type
    aggregated_data = agg_data(args.data, args.dataset_type)

    # Then we convert it to the nnUnet format
    image_label_dict = convert_to_nnUNet_format(aggregated_data, args.output, args.path_data_split, args.task_name, args.task_number, args.dataset_type)
    
    print("Conversion done.")


if __name__ == "__main__":
    main()