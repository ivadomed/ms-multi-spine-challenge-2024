import os
import shutil
import json

# Define the root directory of your dataset and target folders
bids_root = "ms-multi-spine-challenge-2024"
seg_root = os.path.join(bids_root, "derivatives/labels")
output_dir = "dataset_split"

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
        seg_path = os.path.join(seg_root, subject_name, "anat")

        if os.path.exists(subject_path):
            # Copy subject folder
            subject_goal = os.path.join(target_dir, subject_name)
            shutil.copytree(subject_path, subject_goal)

            # Copy segmentation if it exists
            if os.path.exists(seg_path):
                for file in os.listdir(seg_path):
                    if file.endswith(".nii.gz") and "preproc" not in file:
                        seg_file_path = os.path.join(seg_path, file)
                        seg_target_dir = os.path.join(target_dir, subject_name, "anat")
                        os.makedirs(seg_target_dir, exist_ok=True)
                        shutil.copy(seg_file_path, seg_target_dir)

                        if split_key == "training":
                            split_info["training"].append({
                                "image": os.path.join(seg_target_dir, file.replace('_label-lesion_seg','')),
                                "label": os.path.join(seg_target_dir, file)
                            })
                        if split_key == "validation":
                            split_info["validation"].append({
                                "image": os.path.join(seg_target_dir, file.replace('_label-lesion_seg','')),
                                "label": os.path.join(seg_target_dir, file)
                            })
                        elif split_key == "test":
                            split_info["test"].append(os.path.join(seg_target_dir, file.replace('_label-lesion_seg','')))
            else:
                print(f"Segmentation not found for {subject_name}")
        else:
            print(f"Warning: {subject_name} not found in {bids_root}")

# Process each split
for split in splits:
    process_subjects(split["train"], train_dir, "training")
    process_subjects(split["val"], val_dir, "validation")
    process_subjects(split["test"], test_dir, "test")

# Save split info to a JSON file
split_json_path = os.path.join(output_dir, "dataset_split.json")
with open(split_json_path, "w") as json_file:
    json.dump(split_info, json_file, indent=4)

print("Data split completed. Subjects and segmentations copied to train, val, and test folders.")
print(f"Split information saved to {split_json_path}")
