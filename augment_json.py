import os
import json

# Define the root directory of your dataset and label directory
bids_root = "dataset_split"
derivatives_label = os.path.join(bids_root, "derivatives/labels")

# Prepare a dictionary to save the updated dataset info
dataset_info = {
    "description": "ms-multi-spine-challenge-2024-augmented",
    "licence": "custom",
    "modality": {"0": "MRI"},
    "name": "ms-multi-spine-augmented",
    "numTest": 0,
    "numTraining": 0,
    "numValidation": 0,
    "reference": "Custom Dataset",
    "release": "1.0 01/01/2024",
    "tensorImageSize": "3D",
    "training": [],
    "validation": [],
    "test": []
}

# Define the directories for train, validation, and test
train_dir = os.path.join(bids_root, "train")
val_dir = os.path.join(bids_root, "val")
test_dir = os.path.join(bids_root, "test")

# Function to process subjects in a given directory
def process_subjects(subject_dir, dataset_key):
    for subject_folder in os.listdir(subject_dir):
        if subject_folder.startswith("sub-"):
            subject_path = os.path.join(subject_dir, subject_folder,'anat')
            
            if os.path.isdir(subject_path):

                # Loop through files in the subject's directory
                for file in os.listdir(subject_path):
                    if "preproc" not in file : 
                        if file.endswith("T2w.nii.gz"):
                            file_path = os.path.join(subject_path, file)
                            seg_file = file.replace(".nii.gz", "_label-lesion_seg.nii.gz")
                            seg_path = os.path.join(derivatives_label,subject_folder,'anat', seg_file)
                            if os.path.exists(seg_path):
                                
                                dataset_info[dataset_key].append({
                                    "image": file_path,
                                    "label": seg_path
                                })

                        # Process augmented T2w_a files
                        if "T2w_a" in file :
                            file_path = os.path.join(subject_path, file)
                            seg_file = file.replace(".nii.gz", "_seg.nii.gz")
                            seg_path = os.path.join(derivatives_label,subject_folder,'anat', seg_file)
                            
                            if os.path.exists(seg_path):
                                dataset_info[dataset_key].append({
                                    "image": file_path,
                                    "label": seg_path
                                })

# Process subjects in each split
def update_dataset_info():
    process_subjects(train_dir, "training")
    process_subjects(val_dir, "validation")
    process_subjects(test_dir, "test")

    # Update counts
    dataset_info["numTraining"] = len(dataset_info["training"])
    dataset_info["numValidation"] = len(dataset_info["validation"])
    dataset_info["numTest"] = len(dataset_info["test"])

# Update dataset info and save to JSON
update_dataset_info()
output_json_path = os.path.join(bids_root, "augmented_dataset_info.json")
with open(output_json_path, "w") as json_file:
    json.dump(dataset_info, json_file, indent=4)

print(f"Augmented dataset JSON saved to {output_json_path}")
