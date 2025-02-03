import os
import shutil
import json

# Define the root directory of your dataset and target folders
bids_root = "ms-multi-spine-challenge-2024"
seg_root = os.path.join(bids_root, "derivatives")
output_dir = "dataset_split"
output_seg_dir = os.path.join(output_dir, "derivatives")

# Define the target directories for train, validation, and test
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Create the output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define subject split ranges
splits = [
    {"train": range(1, 43), "val": range(43, 50), "test": [50]},
    {"train": range(51, 70), "val": range(70, 75), "test": [75]},
    {"train": range(76, 95), "val": range(95, 100), "test": [100]},
]

# Prepare a dictionary to save the split info
split_info = {
    "description": "ms-multi-spine-challenge-2024",
    "labels": {"0": "background", "1": "vertebrae"},
    "licence": "custom",
    "modality": {"0": "MRI"},
    "name": "ms-multi-spine",
    "numTest": 20,
    "numTraining": 80,
    "reference": "Custom Dataset",
    "release": "1.0 01/01/2024",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],
    "validation": []
}


# Helper function to copy subjects and their segmentations
def process_subjects(subject_range, target_dir, split_key):
    for subject_id in subject_range:
        subject_name = f"sub-{subject_id:03d}"  # Ensure format is sub-001, sub-002, etc.
        subject_path = os.path.join(bids_root, subject_name)
        

        if os.path.exists(subject_path):
            # Copy subject folder
            subject_goal = os.path.join(target_dir, subject_name)
            shutil.copytree(subject_path, subject_goal)

            

# Process each split
for split in splits:
    process_subjects(split["train"], train_dir, "training")
    process_subjects(split["val"], val_dir, "validation")
    process_subjects(split["test"], test_dir, "test")

shutil.copytree(seg_root,output_seg_dir)


