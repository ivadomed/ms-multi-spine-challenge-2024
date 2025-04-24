"""
This script creates test sets for the fine-tuning experiment.
It generates a test set for each contrasts (test only).
It also generates a train set for each contrast (train ).

Input:
    -d: Path to the dataset 150 which contains all the contrasts in the T2w raw space.
    -o: Path to the output directory.

Output:
    None

Usage:
    python make_test_sets.py -d <dataset_dir> -o <output_dir>

Author: Pierre-Louis Benveniste
"""
import os
import json
import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create test sets for fine-tuning experiment.")
    parser.add_argument("-d", "--dataset_dir", type=str, required=True, help="Path to the dataset 150 which contains all the contrasts in the T2w raw space.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the output directory.")
    return parser.parse_args()


def main():

    args = parse_arguments()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # List of contrasts
    contrasts = ["T2w", "STIR", "PSIR", "MP2RAGE"]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Create the test set folder
    test_set_dir = os.path.join(output_dir, "test_sets")
    os.makedirs(test_set_dir, exist_ok=True)
    test_set_t2w_dir = os.path.join(test_set_dir, "T2w")
    os.makedirs(test_set_t2w_dir, exist_ok=True)
    test_set_stir_dir = os.path.join(test_set_dir, "STIR")
    os.makedirs(test_set_stir_dir, exist_ok=True)
    test_set_psir_dir = os.path.join(test_set_dir, "PSIR")
    os.makedirs(test_set_psir_dir, exist_ok=True)
    test_set_mp2rage_dir = os.path.join(test_set_dir, "MP2RAGE")
    os.makedirs(test_set_mp2rage_dir, exist_ok=True)
    # Add the test set labels
    test_set_labels = os.path.join(output_dir, "test_set_labels")
    os.makedirs(test_set_labels, exist_ok=True)

    # Create the train set folder
    train_set_dir = os.path.join(output_dir, "train_sets")
    os.makedirs(train_set_dir, exist_ok=True)
    train_set_t2w_dir = os.path.join(train_set_dir, "T2w")
    os.makedirs(train_set_t2w_dir, exist_ok=True)
    train_set_stir_dir = os.path.join(train_set_dir, "STIR")
    os.makedirs(train_set_stir_dir, exist_ok=True)
    train_set_psir_dir = os.path.join(train_set_dir, "PSIR")
    os.makedirs(train_set_psir_dir, exist_ok=True)
    train_set_mp2rage_dir = os.path.join(train_set_dir, "MP2RAGE")
    os.makedirs(train_set_mp2rage_dir, exist_ok=True)
    # Add the train set labels
    train_set_labels = os.path.join(output_dir, "train_set_labels")
    os.makedirs(train_set_labels, exist_ok=True)

    # Load the conversion
    conv_dict = os.path.join(dataset_dir, "conversion_dict.json")
    with open(conv_dict, "r") as f:
        conv_dict = json.load(f)
    
    # Get all images in the folder imagesTr
    imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
    imagesTr = list(Path(imagesTr_dir).rglob("*.nii.gz"))
    imagesTr = [str(image) for image in imagesTr]
    print(f"Found {len(imagesTr)} images in the folder {imagesTr_dir}")

    for image in imagesTr:
        image_name = None
        for original_image in conv_dict:
            if conv_dict[original_image] == image:
                # print("found")
                image_name = original_image
                break
        # Define new image name
        out_image = image_name.split("/")[-1].split("_")[0]
        contrast = image_name.split("/")[-1].split("_")[-1].split(".")[0]
        out_image = os.path.join(train_set_dir, contrast, out_image + ".nii.gz")
        # Copy the image to the new location
        os.system(f"cp {image} {out_image}")
        # At the same time, we copy the corresponding label to the labels_folder
        label_file = os.path.join(dataset_dir, "labelsTr", image.split("/")[-1].replace("_0000",""))
        out_label = os.path.join(train_set_labels, image_name.split("/")[-1].split("_")[0] + ".nii.gz")
        # Copy the label to the new location
        os.system(f"cp {label_file} {out_label}")

    # Same for the test set
    imagesTs_dir = os.path.join(dataset_dir, "imagesTs")
    imagesTs = list(Path(imagesTs_dir).rglob("*.nii.gz"))
    imagesTs = [str(image) for image in imagesTs]
    print(f"Found {len(imagesTs)} images in the folder {imagesTs_dir}")
    for image in imagesTs:
        image_name = None
        for original_image in conv_dict:
            if conv_dict[original_image] == image:
                # print("found")
                image_name = original_image
                break
        # Define new image name
        out_image = image_name.split("/")[-1].split("_")[0]
        contrast = image_name.split("/")[-1].split("_")[-1].split(".")[0]
        out_image = os.path.join(test_set_dir, contrast, out_image + ".nii.gz")
        # Copy the image to the new location
        os.system(f"cp {image} {out_image}")
        # At the same time, we copy the corresponding label to the labels_folder
        label_file = os.path.join(dataset_dir, "labelsTs", image.split("/")[-1].replace("_0000",""))

        out_label = os.path.join(test_set_labels, image_name.split("/")[-1].split("_")[0] + ".nii.gz")
        # Copy the label to the new location
        os.system(f"cp {label_file} {out_label}")
    
    print("Done creating test sets.")
    

if __name__ == "__main__":
    main()