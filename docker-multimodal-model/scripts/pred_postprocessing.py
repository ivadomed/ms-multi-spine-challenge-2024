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
import numpy as np 
import nibabel as nib 


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocessing script")
    parser.add_argument("-i", "--predicted_segmentation", type=str, required=True, help="Path to the predicted segmentation")
    parser.add_argument("-t", "--type", type=str, required=True, help="Type of contrast among PSIR STIR MP2RAGE")
    parser.add_argument("-d", "--subj_dict", type=dict, required=True, help="Dictionary with the paths of the images of the subject")
    return parser.parse_args()


def majority_vote(volumes, t):
    stacked = np.stack(volumes, axis=0)  # Shape: (5, H, W, D)
    vote_sum = np.sum(stacked, axis=0)
    if t == "STIR":
        return (vote_sum > 0.5).astype(np.uint8), (vote_sum/5)
    else : 
        return (vote_sum > 2.5).astype(np.uint8), (vote_sum/5)

def postprocess_segmentation(subj_dict):
    # Build a temp folder in the output folder
    temp_folder = 'temp_postprocessing'
    os.makedirs(temp_folder, exist_ok=True)

    # Retrieve the segmentations for the five folds: 

    predicted_segmentation = subj_dict['seg_path']

    subject_files = sorted([
        f for f in os.listdir(os.path.join(predicted_segmentation, "fold_0", "prediction"))
        if f.endswith(".nii.gz")
    ])

    # Postprocessing running here

    print(f"Found {len(subject_files)} subjects to process...")
    
    # We send the predicted segmentation to the T2w raw image space
    t2w_raw_image = subj_dict['rawdata_T2']
        
    

    for subject_file in subject_files:
        predictions = []
        affine, header = None, None

        for fold in range(5):
            fold_pred_path = os.path.join(predicted_segmentation, f"fold_{fold}", "prediction", subject_file)

            assert os.system(f"sct_register_multimodal -i {fold_pred_path} -d {t2w_raw_image} -identity 1 -o {fold_pred_path}  ") == 0


            if not os.path.exists(fold_pred_path):
                raise FileNotFoundError(f"Missing prediction for {subject_file} in fold {fold}")

            pred_nib = nib.load(fold_pred_path)
            pred_data = pred_nib.get_fdata()


            if affine is None:
                affine = pred_nib.affine
                header = pred_nib.header

            predictions.append(pred_data)

        # ✅ Apply correct majority vote over 5 folds
        voted_mask, soft_mask = majority_vote(predictions, subj_dict['t'])

        out_path = os.path.join(subj_dict['seg_path'], subject_file.replace('.nii.gz', '_voted_mask.nii.gz'))
        nib.save(nib.Nifti1Image(voted_mask, affine, header), out_path)

        subj_dict['segmentation_file'] = out_path

        # Save the soft mask as well
        soft_mask_path = os.path.join(subj_dict['seg_path'], subject_file.replace('.nii.gz', '_soft_mask.nii.gz'))
        nib.save(nib.Nifti1Image(soft_mask, affine, header), soft_mask_path)

        subj_dict['soft_mask'] = soft_mask_path

        print(f"Saved voted mask for {subject_file}")


    
    return subj_dict


if __name__ == "__main__":
    args = parse_args()
    postprocessed_segmentation = postprocess_segmentation(args.subj_dict)

