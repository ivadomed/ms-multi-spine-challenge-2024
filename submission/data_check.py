"""
Script dedicated to data consistency verification, based on the fact that
the proposed pipelines are required to output:

- a consistent nii.gz file with dimension and orientation matrix equals to
those of the input t2 sagital acquisitions (the spatial coverage of the
sagittal T2 will always be the one considered for annotating the lesions and
evaluating the methods). This nii.gz file must contain only integer values
where each lesion of a given instance in a given case is assigned to a unique
integer value (≥1, 0 being the background).

- a consistent csv file with the two columns : id and p and one row per
instance in the corresponding nii.gz file. id must match a instance number
from the nii file and p must be a float in [0, 1].

This script takes as input the input T2Sag and the corresponding output data
and checks that they are consistent with the rules described above.
The success of this script is required to ensure the compatibility of the method
output to the overall evaluation process.
"""


import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import json


def check_image_properties(ref_image, pred_image):
    """
    Check if the reference and predicted images have the same orientation, spacing, and dimensions.
    Allow a tolerance of e-5 for comparing direction matrices.
    """
    # Check spacing
    if ref_image.GetSpacing() != pred_image.GetSpacing():
        raise ValueError("Reference and predicted images have different spacings.")
    
    # Check dimensions
    if ref_image.GetSize() != pred_image.GetSize():
        raise ValueError("Reference and predicted images have different dimensions.")
    
    # Check direction with tolerance
    ref_direction = np.array(ref_image.GetDirection()).reshape(3, 3)
    pred_direction = np.array(pred_image.GetDirection()).reshape(3, 3)
    
    if not np.allclose(ref_direction, pred_direction, atol=1e-3):
        raise ValueError("Reference and predicted images have different orientations.")


def check_predicted_values(pred_image):
    """
    Check if the predicted segmentation image contains only integer values.
    """
    pred_array = sitk.GetArrayFromImage(pred_image)
    if not np.all(np.equal(pred_array, np.floor(pred_array))):  # Check if all values are integers
        raise ValueError("Predicted image contains non-integer values.")


def check_csv_values(csv_path):
    """
    Check that the 'label' column contains only integers and the 'p' column only contains values between 0 and 1.
    """
    df = pd.read_csv(csv_path)

    # Check 'label' column for integers
    if not np.all(df['label'].apply(lambda x: isinstance(x, int))):
        raise ValueError(f"CSV file {csv_path} has non-integer values in the 'label' column.")

    # Check 'p' column for probabilities between 0 and 1
    if not np.all(df['p'].apply(lambda x: 0 <= x <= 1)):
        raise ValueError(f"CSV file {csv_path} has values outside [0, 1] in the 'p' column.")


def main(args):
    """
    Validate the images and CSV files for the specified folders and store results in JSON.
    """
    ref_files = sorted([f for f in os.listdir(args.ref_folder) if '.nii' in f])
    pred_files = sorted([f for f in os.listdir(args.pred_folder) if '.nii' in f])
    csv_files = sorted([f for f in os.listdir(args.csv_folder) if f.endswith('.csv')])

    results = {
        "Number of files": "ok",
        "Image properties": [],
        "Predicted image values": [],
        "CSV values": []
    }

    if len(ref_files) != len(pred_files) or len(ref_files) != len(csv_files):
        results["Number of files"] = "Mismatch in the number of reference files, predicted files, and CSV files."

    for ref_file in ref_files:
        pred_file = next((f for f in pred_files if f == ref_file), 'None')
        csv_file = next((f for f in csv_files if f == ref_file.replace('.nii.gz', '.csv')), 'None')
        ref_path = os.path.join(args.ref_folder, ref_file)
        pred_path = os.path.join(args.pred_folder, pred_file)
        csv_path = os.path.join(args.csv_folder, csv_file)

        # Initialize result entries
        image_properties_result = {
            "ref_file": ref_file,
            "pred_file": pred_file,
            "csv_file": csv_file,
            "error": "ok"
        }
        predicted_values_result = {
            "pred_file": pred_file,
            "error": "ok"
        }
        csv_values_result = {
            "csv_file": csv_file,
            "error": "ok"
        }

        # Validate image properties
        try:
            if not os.path.exists(pred_path) or pred_file == 'None':
                raise ValueError(f"Predicted image file {pred_file} does not exist.")
            ref_image = sitk.ReadImage(ref_path)
            pred_image = sitk.ReadImage(pred_path)
            check_image_properties(ref_image, pred_image)
        except ValueError as e:
            image_properties_result["error"] = f"Error with images properties {ref_file} and {pred_file}: {e}"

        # Validate predicted image values
        try:
            if not os.path.exists(pred_path) or pred_file == 'None':
                raise ValueError(f"Predicted image file {pred_file} does not exist.")
            check_predicted_values(pred_image)
        except ValueError as e:
            predicted_values_result["error"] = f"Error with predicted image content {pred_file}: {e}"

        # Validate CSV values
        try:
            if not os.path.exists(csv_path) or csv_file == 'None':
                raise ValueError(f"CSV file {csv_file} does not exist.")
            check_csv_values(csv_path)
        except ValueError as e:
            csv_values_result["error"] = f"Error with CSV file content {csv_file}: {e}"

        # Append results
        results["Image properties"].append(image_properties_result)
        results["Predicted image values"].append(predicted_values_result)
        results["CSV values"].append(csv_values_result)

    # Save results to JSON file
    output_json_path = os.path.join(args.output_folder, "validation_results.json")
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    if args.debug:
        print(f"Validation results saved to {output_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate T2 sagittal image outputs for consistency."
    )
    parser.add_argument('-r', '--ref_folder', required=True, help="Path to the reference images folder.")
    parser.add_argument('-p', '--pred_folder', required=True, help="Path to the predicted images folder.")
    parser.add_argument('-c', '--csv_folder', required=True, help="Path to the CSV files folder.")
    parser.add_argument('-o', '--output_folder', required=True, help="Path to save the validation results JSON.")
    parser.add_argument('-debug', '--debug', action='store_true', help='Prints some information.')

    args = parser.parse_args()

    main(args)
