"""
This script converts the data to the nnUNet format. 
It asks if the data wanted is the registered or the unregistered data.
Script is inspired from: https://github.com/ivadomed/ms-lesion-agnostic/blob/main/nnunet/convert_msd_to_nnunet_reorient.py

Input: 
    -i: Path to the input data
    -o: Path to the output data
    -r: If the data is registered or not
    -t: Task number

Output:
    None

Example:
    python convert_to_nnunet_format.py -i /path/to/input/data -o /path/to/output/data -t 101

Author: Pierre-Louis Benveniste
"""

import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from collections import OrderedDict
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Convert data to nnUNet format')
    parser.add_argument('-i', '--input', help='Path to the input data', required=True)
    parser.add_argument('-o', '--output', help='Path to the output data', required=True)
    parser.add_argument('-r', '--registered', help='If the data is registered or not', action='store_true')
    parser.add_argument('-t', '--tasknumber', help='Task number', required=True)
    args = parser.parse_args()
    return args


def main():

    # Get arguments
    args = parse_args()
    data_path = args.input
    output_path = args.output

    taskName = "MsMultiSpine"

    # Initialize the seed numpy
    np.random.rand(42)

    # Define the output path
    path_out = Path(os.path.join(output_path, f'Dataset{args.tasknumber}_{taskName}'))

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

    # Image and label dictionaries
    image_label_dict = {}
    
    # We get all the data in the input folder
    ## If registered data, for each image we have the corresponding label
    if args.registered:
        # Get the list of all derivatives files using rglob
        derivatives = list(Path(data_path).rglob('*desc-preprocReg_T2w_label-lesion_seg.nii.gz'))
        # We iterate over the derivatives and get the corresponding image
        for derivative in derivatives:
            image = derivative.name.replace('T2w_label-lesion_seg.nii.gz', '')
            # Get all images with the same name
            images = list(Path(data_path).rglob(f'*{image}*.nii.gz'))
            # Remove derivatives 
            images = [x for x in images if 'derivatives' not in str(x)]
            # For each image we add it to the dictionary
            for img in images:
                image_label_dict[str(img)] = str(derivative)

    else:
        # Get the list of all derivatives files using rglob
        derivatives = list(Path(data_path).rglob('*label-lesion_seg.nii.gz'))
        # Remove the registered data and the preprocessed data
        derivatives = [x for x in derivatives if 'preproc' not in str(x)]
        # In this case, we only have one image for each label
        for derivative in derivatives:
            # Get the corresponding image
            image = str(derivative).replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('/derivatives/labels', '')
            # Add the image and the label to the dictionary
            image_label_dict[str(image)] = str(derivative)    

    # We iterate over the images and labels of the dictionary to build the nnUNet format dataset
    for image, label in image_label_dict.items():
        # If random number is above 0.8, we put the image in the test set
        if np.random.rand() > 0.8:
            scan_cnt_test += 1
            # Define the image and label files
            image_file_nnunet = os.path.join(path_out_imagesTs,f'{taskName}_{scan_cnt_test:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,f'{taskName}_{scan_cnt_test:03d}.nii.gz')
            # Add image to test list
            test_images.append(str(image_file_nnunet))
            test_labels.append(str(label_file_nnunet))

        else:
            scan_cnt_train += 1
            # Define the image and label files
            image_file_nnunet = os.path.join(path_out_imagesTr,f'{taskName}_{scan_cnt_train:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTr,f'{taskName}_{scan_cnt_train:03d}.nii.gz')
            # Add image to train list
            train_images.append(str(image_file_nnunet))
            train_labels.append(str(label_file_nnunet))
        
        # Instead of copying we will reorient the image to RPI
        assert os.system(f"sct_image -i {image} -setorient RPI -o {image_file_nnunet}") ==0
        # Binarize the label and save it to the adequate path
        label_data = nib.load(label).get_fdata()
        label_data[label_data > 0] = 1
        label_file = nib.Nifti1Image(label_data, nib.load(label).affine)
        nib.save(label_file, label_file_nnunet)
        # Then we reorient the label to RPI
        assert os.system(f"sct_image -i {label_file_nnunet} -setorient RPI -o {label_file_nnunet}") ==0

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(image))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(label))] = label_file_nnunet

        # For each label files, we reorient them to the same orientation as the image, using sct_register_multimodal -identity 1
        assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") ==0
        # Remove the other useless files
        assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") ==0
        other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
        assert os.system(f"rm {other_file_to_remove}") ==0

        # Then we binarize the label
        assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") ==0

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
    json_dict['name'] = taskName
    json_dict['description'] = taskName
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    
    # Because only using one modality  
    ## was changed from 'modality' to 'channel_names'
    json_dict['channel_names'] = {
            "0": "MRI",
        }
    
     # 0 is always the background. Any class labels should start from 1.
    json_dict['labels'] = {
        "background" : 0,
        "lesion" : 1,
    }
   
    # json_dict['regions_class_order'] = [1,2]

    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test
    #Newly required field in the json file with v2
    json_dict["file_ending"] = ".nii.gz"

    json_dict['training'] = [{'image': str(train_labels[i]).replace("labelsTr", "imagesTr") , "label": train_labels[i] }
                                 for i in range(len(train_images))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described

    #Removed because useless in this case
    json_dict['test'] = [{'image': str(test_labels[i]).replace("labelsTs", "imagesTs") , "label": test_labels[i] }
                                 for i in range(len(test_images))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)

    return None


if __name__ == '__main__':
    main()