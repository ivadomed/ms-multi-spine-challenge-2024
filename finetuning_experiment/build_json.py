"""
This script is used to build json files for inference

"""

path_dataset = "/home/plbenveniste/net/challenge-multi-spine/data/ms-multi-spine-challenge-2024/"

json_path_out = "/home/plbenveniste/net/challenge-multi-spine/finetuning_experiment/ms-multi-spine-challenge-2024/finetuning_experiment/images_dict.json"

data_split = "/home/plbenveniste/net/challenge-multi-spine/data_preproc/ms-multi-spine-challenge-2024/data_preproc/data_split.yml"

import json
from pathlib import Path
import yaml
import os

# List all files in the dataset which we will use: 
images = Path(path_dataset).rglob("*desc-preproc*.nii.gz")
# Remove all images with desc-preprocReg in name
images = [str(img) for img in images if "desc-preprocReg" not in str(img)]
# Remove all derivative files
images = [str(img) for img in images if "derivatives" not in str(img)]

# Load the data split
with open(data_split, 'r') as file:
    data_split_yml = yaml.load(file, Loader=yaml.FullLoader)
training_subjects = data_split_yml['TRAINING']
testing_subjects = data_split_yml['TESTING']

# For each image, we build the following dictionnary:
## input_image, corresponding T2w_raw image, corresponding T2w raw lesion mask, contrast
training_dict = []
test_dict = []

# Iterate over the images
for file in images:
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
    subj_dict = {
        'imge_name': file.split('/')[-1],
        'subject_name': subject_name,
        'input_image': file,
        't2w_raw_label_file': label_file,
        't2w_raw_image': t2w_raw_image,
        'contrast': contrast
    }
    if subject_name in training_subjects:
        training_dict.append(subj_dict)
    else:
        test_dict.append(subj_dict)

# Build final dictionnary
final_dict = {
    'training': training_dict,
    'testing': test_dict
}
# Save the final dict to json
with open(json_path_out, 'w') as json_file:
    json.dump(final_dict, json_file, indent=4)

print(f"Json file saved to {json_path_out}")