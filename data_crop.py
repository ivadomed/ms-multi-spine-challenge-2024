import os
import json
import nibabel as nib
import numpy as np

def get_nonzero_bbox(image_data):
    nonzero_coords = np.argwhere(image_data > 0)
    min_idx = np.min(nonzero_coords, axis=0)
    max_idx = np.max(nonzero_coords, axis=0) + 1  # Include last index
    return tuple(slice(min_idx[d], max_idx[d]) for d in range(image_data.ndim)), min_idx

def crop_and_save_nifti(input_path, output_path, bbox, min_idx):
    nifti = nib.load(input_path)
    cropped_data = nifti.get_fdata()[bbox]
    new_affine = nifti.affine.copy()
    new_affine[:3, 3] += nifti.affine[:3, :3] @ min_idx  # Update origin
    cropped_nifti = nib.Nifti1Image(cropped_data, affine=new_affine, header=nifti.header)
    cropped_nifti.header.set_data_shape(cropped_data.shape)  # Update header with new shape
    nib.save(cropped_nifti, output_path)

def convert_to_nnUNet_format(root_dir, output_dir, task_old="Dataset030_ms-challenge-multimodal", task_new="Dataset031_ms-challenge-multimodal-cropped"):
    nnUNet_base_old = os.path.join(output_dir, "nnUNet_raw", task_old)
    nnUNet_base_new = os.path.join(output_dir, "nnUNet_raw", task_new)
    
    imagesTr_old = os.path.join(nnUNet_base_old, "imagesTr")
    labelsTr_old = os.path.join(nnUNet_base_old, "labelsTr")
    imagesTr_new = os.path.join(nnUNet_base_new, "imagesTr")
    labelsTr_new = os.path.join(nnUNet_base_new, "labelsTr")
    imagesTs_new = os.path.join(nnUNet_base_new, "imagesTs")
    
    os.makedirs(imagesTr_new, exist_ok=True)
    os.makedirs(imagesTs_new, exist_ok=True)
    os.makedirs(labelsTr_new, exist_ok=True)
    
    subjects = [f.split("_")[0] for f in os.listdir(imagesTr_old) if f.endswith("_0000.nii.gz")]
    
    for sub in subjects:
        t2w_path_old = os.path.join(imagesTr_old, f"{sub}_0000.nii.gz")
        contrast_path_old = os.path.join(imagesTr_old, f"{sub}_0001.nii.gz")
        label_path_old = os.path.join(labelsTr_old, f"{sub}.nii.gz")
        
        t2w_nifti = nib.load(t2w_path_old)
        bbox, min_idx = get_nonzero_bbox(t2w_nifti.get_fdata())
        
        crop_and_save_nifti(t2w_path_old, os.path.join(imagesTr_new, f"{sub}_0000.nii.gz"), bbox, min_idx)
        crop_and_save_nifti(contrast_path_old, os.path.join(imagesTr_new, f"{sub}_0001.nii.gz"), bbox, min_idx)
        if os.path.exists(label_path_old):
            crop_and_save_nifti(label_path_old, os.path.join(labelsTr_new, f"{sub}.nii.gz"), bbox, min_idx)
    
    with open(os.path.join(nnUNet_base_old, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    with open(os.path.join(nnUNet_base_new, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Cropped dataset saved in {nnUNet_base_new}")

# Example usage:
convert_to_nnUNet_format("nnUNet", "nnUNet")
