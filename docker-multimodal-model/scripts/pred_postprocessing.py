"""
In this script we perform postprocessing of the predicted segmentations. 

Input: 
    -predicted-segmentation: path to the predicted segmentation
    -subj-dict: dictionary with the paths of the images of the subject

Returns:
    -postprocessed-segmentation: path to the postprocessed segmentation
"""
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocessing script")
    parser.add_argument("-i", "--predicted_segmentation", type=str, required=True, help="Path to the predicted segmentation")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    return parser.parse_args()


def postprocess_segmentation(predicted_segmentation, subj_dict, t2w_raw_image):
    # Build a temp folder in the output folder
    temp_folder = 'temp_postprocessing'
    os.makedirs(temp_folder, exist_ok=True)

    # Postprocessing running here
    
    # We send the predicted segmentation to the T2w raw image space
    assert os.system(f"sct_register_multimodal -i {predicted_segmentation} -d {t2w_raw_image} -identity 1 -ofolder {temp_folder} -owarp {temp_folder/'warp_image_to_t2wraw1.nii.gz'} -o {temp_folder/'image_reg_to_t2wraw1.nii.gz'}  -dseg {temp_folder/'raw_t2w_sc_seg.nii.gz'} ") == 0



    # Build the postprocessed segmentation path (output folder and image name)
    postprocessed_segmentation = os.path.join(temp_folder, Path(predicted_segmentation).name)
    return postprocessed_segmentation


if __name__ == "__main__":
    args = parse_args()
    postprocessed_segmentation = postprocess_segmentation(args.predicted_segmentation, args.subj_dict)
