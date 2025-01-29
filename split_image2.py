import os
import json

# Define the root directory of your dataset split
dataset_root = "dataset_split"
seg_root = os.path.join(dataset_root,"derivatives","labels")

# Define the modalities to search for image2
modalities = ["PSIR", "STIR", "MP2RAGE"]

# Prepare a dictionary to save the split info
split_info = {
    "description": "ms-multi-spine-challenge-2024",
    "labels": {"0": "background", "1": "vertebrae"},
    "licence": "custom",
    "modality": {"0": "MRI"},
    "name": "ms-multi-spine",
    "reference": "Custom Dataset",
    "release": "1.0 01/01/2024",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],
    "validation": [],
}

# Helper function to populate the JSON entries
def process_split(split_folder, split_key):
    split_dir = os.path.join(dataset_root, split_folder)
    
    if not os.path.exists(split_dir):
        print(f"Warning: {split_folder} folder not found in {dataset_root}")
        return

    for subject_name in os.listdir(split_dir):
        subject_path = os.path.join(split_dir, subject_name, "anat")
        subject_seg_path = os.path.join(seg_root, subject_name, "anat")
        if not os.path.exists(subject_path):
            print(f"Warning: anat folder not found for {subject_name} in {split_folder}")
            continue

        # Find T2 image
        t2_image = None
        for file in os.listdir(subject_path):
            if "T2" in file and file.endswith(".nii.gz") and "preproc" not in file and "seg" not in file:
                t2_image = os.path.join(subject_path, file)
                break

        if not t2_image:
            print(f"T2 image not found for {subject_name} in {split_folder}")
            continue

        # Find image2 (PSIR, STIR, MP2RAGE)
        image2 = None
        for modality in modalities:
            for file in os.listdir(subject_path):
                if modality in file and file.endswith(".nii.gz") and "preproc" not in file:
                    image2 = os.path.join(subject_path, file)
                    break
            if image2:
                break

        if not image2:
            print(f"Second image not found for {subject_name} in {split_folder}")
            continue

        # Find segmentation
        seg_file = None
        for file in os.listdir(subject_seg_path):
            if file.endswith(".nii.gz") and "label-lesion_seg" in file and "preproc" not in file :
                seg_file = os.path.join(subject_seg_path, file)
                break

        if not seg_file:
            print(f"Segmentation not found for {subject_name} in {split_folder}")
            continue

        # Add to the appropriate split in JSON
        entry = {
            "image1": t2_image,
            "image2": image2,
            "label": seg_file,
        }
        split_info[split_key].append(entry)

# Process each split folder
splits = {
    "train": "training",
    "val": "validation",
    "test": "test",
}

for folder, split_key in splits.items():
    process_split(folder, split_key)

# Save the split info to a JSON file
split_json_path = os.path.join(dataset_root, "dataset_split.json")
with open(split_json_path, "w") as json_file:
    json.dump(split_info, json_file, indent=4)

print(f"Dataset split JSON created at {split_json_path}")
