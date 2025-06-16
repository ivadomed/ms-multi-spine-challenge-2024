"""
This script runs the evaluation of the predictions. It uses the script evaluate_semantic_pred.py in the evaluation branch.

Input: 
    -pred-folder: The folder containing the predictions.
    -image-dict: The dictionary containing the image names and their corresponding paths.
    -output-folder: The folder where the evaluation results will be saved.
    -evaluation-script: The script used for evaluation.

Output:
    None

Example:
    python run_evaluations.py --pred-folder /path/to/predictions --image-dict /path/to/image_dict.json --output-folder /path/to/output --evaluation-script /path/to/evaluate_semantic_pred.py

Author: Pierre-Louis Benveniste
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluations on predictions.")
    parser.add_argument("--pred-folder", type=str, required=True, help="Path to the folder containing the predictions.")
    parser.add_argument("--image-dict", type=str, required=True, help="Path to the image dictionary file.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--evaluation-script", type=str, required=True, help="Path to the evaluation script.")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    pred_folder = args.pred_folder
    image_dict_path = args.image_dict
    output_folder = args.output_folder
    evaluation_script = args.evaluation_script

    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the json file
    with open(image_dict_path, "r") as f:
        image_dict = json.load(f)

    # We consider that for now, we only focus on testing images
    images = image_dict['testing']

    # List all the predictions
    pred_files = list(Path(pred_folder).rglob("*.nii.gz"))

    # Initiate a array of the results
    results = {}

    # Iterate over files:
    for pred_file in tqdm(pred_files):
        # Get the image name
        image_name = os.path.basename(pred_file).split('.')[0]
        
        # Get the corresponding label
        label = images[image_name]['t2w_raw_label_file']

        # Run the evaluation script
        assert os.system(f"python {evaluation_script} -pred {pred_file} -label {label} -output {output_folder} -image-name {image_name}") == 0

        # Load the results
        with open(os.path.join(output_folder, f"{image_name}.json"), "r") as f:
            result = json.load(f)

        results[image_name] = result[image_name]

    # Save each metric in a separate file
    # Save the results
    with open(os.path.join(output_folder, "dice_scores.txt"), "w") as f:
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['dice']}\n")
    with open(os.path.join(output_folder, "ppv_scores.txt"), "w") as f:
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['ppv']}\n")
    with open(os.path.join(output_folder, "f1_scores.txt"), "w") as f:
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['f1']}\n")
    with open(os.path.join(output_folder, "sensitivity_scores.txt"), "w") as f:
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['sensitivity']}\n")
    with open(os.path.join(output_folder, "nsd_scores.txt"), "w") as f:
        for image_name, result in results.items():
            f.write(f"{image_name}: {result['nsd']}\n")

    # In a txt file save the average and std of each results
    with open(os.path.join(output_folder, "scores_summary.txt"), "w") as f:
        f.write(f"Dice score: {sum([result['dice'] for result in results.values()]) / len(results)} ± {np.std([result['dice'] for result in results.values()])}\n")
        f.write(f"PPV score: {sum([result['ppv'] for result in results.values()]) / len(results)} ± {np.std([result['ppv'] for result in results.values()])}\n")
        f.write(f"F1 score: {sum([result['f1'] for result in results.values()]) / len(results)} ± {np.std([result['f1'] for result in results.values()])}\n")
        f.write(f"Sensitivity score: {sum([result['sensitivity'] for result in results.values()]) / len(results)} ± {np.std([result['sensitivity'] for result in results.values()])}\n")
        f.write(f"NSD score: {sum([result['nsd'] for result in results.values()]) / len(results)} ± {np.std([result['nsd'] for result in results.values()])}\n")
    

if __name__ == "__main__":
    main()