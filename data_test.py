import os
import shutil
import json
import nibabel as nib
import numpy as np

def convert_to_nnUNet_format(root_dir, output_dir, task_name="Dataset240_ms-challenge-multimodal"):
    nnUNet_base = os.path.join(output_dir, "nnUNet_raw", task_name)
    imagesTr = os.path.join(nnUNet_base, "imagesTr")
    labelsTr = os.path.join(nnUNet_base, "labelsTr")
    imagesTs = os.path.join(nnUNet_base, "imagesTs")
    
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    
    subjects = [d for d in os.listdir(root_dir) if d.startswith("sub-")]
    
    contrast_list = ["STIR", "PSIR", "MP2RAGE"]
    
    for sub in subjects:
        anat_dir = os.path.join(root_dir, sub, "anat")
        label_dir = os.path.join(root_dir, "derivatives", "labels", sub, "anat")
        
        t2w_image = os.path.join(anat_dir, f"{sub}_desc-preprocReg_T2w.nii.gz")
        label_image = os.path.join(label_dir, f"{sub}_desc-preprocReg_T2w_label-lesion_seg.nii.gz")
        
        assert os.system(f"sct_image -i {t2w_image} -setorient RPI -o {imagesTr}/{sub}_0000.nii.gz") ==0

       
       
        
        if os.path.exists(label_image):
            label_nifti = nib.load(label_image)
            label_data = label_nifti.get_fdata()
            
            # Convert labels: set non-zero voxels to 1, keep zeros as 0
            binary_label_data = (label_data > 0).astype(np.int8)
            
            new_label_nifti = nib.Nifti1Image(binary_label_data, affine=label_nifti.affine, header=label_nifti.header)
            label_path = os.path.join(labelsTr, f"{sub}.nii.gz")
            nib.save(new_label_nifti, label_path)
            assert os.system(f"sct_image -i {label_path} -setorient RPI -o {label_path}") ==0
    
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(subjects),
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(nnUNet_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Dataset successfully converted to nnUNet format at {nnUNet_base}")

# Example usage:
convert_to_nnUNet_format("ms-multi-spine-challenge-2024", "nnUNet")
