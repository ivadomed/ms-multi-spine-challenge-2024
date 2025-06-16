# Data preprocessing

The scripts in this folder perform data preprocessing. 

The scripts: 
- [convert_to_nnunet.py](convert_to_nnunet.py): This script is used to convert the ms-multi-spine challenge dataset from the BIDS format to the nnUNet format. The script will need to create the types of dataset detailed in the PR head: https://github.com/ivadomed/ms-multi-spine-challenge-2024/pull/13. It takes as input a predefined data split (detailed in this issue: https://github.com/ivadomed/ms-multi-spine-challenge-2024/issues/12). 
- [convert_to_nnunet_T2w_raw_space.py](convert_to_nnunet_T2w_raw_space.py): This script is used to generated nnUNet format dataset from the original dataset. More precisely, it creates the dataset 11, 12, 13, 14 and 15: more details here: https://github.com/ivadomed/ms-multi-spine-challenge-2024/pull/13#issuecomment-2805024837. With this script, all images are stored in the T2w raw space. 
- [convert_to_nnunet_contrast_raw_space.py](convert_to_nnunet_contrast_raw_space.py): This script is used to generated nnUNet format dataset from the original dataset. More precisely, it creates the dataset 21, 22, 23, 24 and 25. 21: T2w raw space cropped masked around the spinal cord. 22: PSIR raw space cropped masked around the spinal cord. 23: STIR raw space cropped masked around the spinal cord. 24: MP2RAGE raw space cropped masked around the spinal cord. 25: all raw space cropped masked around the spinal cord. 

The script used for the the final submission was `convert_to_nnunet_T2w_raw_space.py` for both the monomodal and the multimodal models.

The `data_split.yml` file was done manually to split the dataset between training and testing per subject. It was done so that, the distribution of contrasts in the train set and in the test set are equal. 