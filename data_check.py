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

This script takes as input the input T2Sag (in input_folder) and the corresponding output data (in output_folder)
and checks that they are consistent with the rules described above.
The success of this script is required to ensure the compatibility of the method
output to the overall evaluation process.

If all files are present in input_folder and output_folder with correct name for input_folder 
a json file "validation_results" will be created in output_folder with more information about data checking.
Else a message in the console will be printed to inform what is the error with input_folder and output_folder.
"""

import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import json

from listing_inputs import listing_input_files


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


def check_csv_values(csv_path, pred_image):
    """
    Check that the 'id' column contains only integers and the 'p' column only contains values between 0 and 1.
    Also check if same lesion id in pred_image and the csv file
    """
    df = pd.read_csv(csv_path)
    pred_array = sitk.GetArrayFromImage(pred_image)

    # Check 'id' column for integers
    if not np.all(df['id'].apply(lambda x: isinstance(x, int))):
        raise ValueError(f"CSV file {csv_path} has non-integer values in the 'id' column.")

    # Check 'p' column for probabilities between 0 and 1
    if not np.all(df['p'].apply(lambda x: 0 <= x <= 1)):
        raise ValueError(f"CSV file {csv_path} has values outside [0, 1] in the 'p' column.")

    # Check label in pred_image and label in csv file 
    ids_pred_image = [int(id) for id in np.unique(pred_array) if id > 0]
    ids_csv = df["id"].unique()

    if len(ids_pred_image) != len(ids_csv):
        raise ValueError(f"Different number of lesion in predicted image and CSV file.")


def main(args):
    """
    Validate the images and CSV files for the specified folders and store results in JSON.
    """

    # Input folder : 
    input_folder = args.input_folder
    # Verifying if input_folder exists : 
    if not(os.path.isdir(input_folder)):
        print(f"The input folder {input_folder} does not exist.")
        return

    # Output folder : 
    output_folder = args.output_folder
    # Verifying if output_folder exists : 
    if not(os.path.isdir(output_folder)):
        print(f"The output folder {output_folder} does not exist.")
        return

    # Listing all files in output_folder :
    output_folder_files = os.listdir(output_folder)
    
    # CSV file name and path, Segmentation mask name and path :
    # Getting all CSV files in output_folder :
    csv_file = [file for file in output_folder_files if file.endswith('.csv')]
    # Verifying that there is only one CSV file in output_folder :
    if len(csv_file) != 1:
        print(f"Incorrect number of CSV file in {output_folder}, we must only have one CSV file.")
        return
    else:
        csv_file = csv_file[0]
        csv_file_path = os.path.join(output_folder, csv_file)
    
    # Getting all nifti files in output_folder :
    pred_file = [file for file in output_folder_files if file.endswith('.nii.gz')]
    # Verifying that there is only one nifti file in output_folder :
    if len(pred_file) != 1:
        print(f"Incorrect number of segmentation mask in {output_folder}, we must only have one segmentation file.")
        return
    else:
        pred_file = pred_file[0]
        pred_file_path = os.path.join(output_folder, pred_file)

    # Getting path to all files in input_folder with a pattern corresponding to the one in the given data :
    ref_files_path = listing_input_files(input_folder)

    # Verifying that the name of files in input_folder is corresponding to name of files in the given data :
    if len(ref_files_path) == 0:
        print(f"No files in {input_folder} with same name pattern as in the input folder.")
        return

    # Getting path to the raw T2 sagital :
    ref_file_path = [file for file in ref_files_path if file and 'T2' in file and 'rawdata' in file]
    
    # Verifying that the raw T2 sagital exists : 
    if len(ref_file_path) != 1:
        print(f"No raw T2 sagital image in {input_folder} or the name is not corresponding to the pattern.")
        return
    else:
        ref_file_path = ref_file_path[0]
        ref_file = os.path.basename(ref_file_path)

    results = {
        # "Number of files": "ok",
        "Image properties": [],
        "Predicted image values": [],
        "CSV values": []
    }

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
        if not os.path.exists(pred_file_path) or pred_file == 'None':
            raise ValueError(f"Predicted image file {pred_file} does not exist.")
        ref_image = sitk.ReadImage(ref_file_path)
        pred_image = sitk.ReadImage(pred_file_path)
        check_image_properties(ref_image, pred_image)
        print('Check image properties ok !')

    except ValueError as e:
        image_properties_result["error"] = f"Error with images properties {ref_file} and {pred_file}: {e}"
        print(f"Check image properties failed !\nLook at {os.path.join(output_folder,'validation_results.json')} for more information !")

    # Validate predicted image values
    try:
        if not os.path.exists(pred_file_path) or pred_file == 'None' or pred_image == 'None':
            raise ValueError(f"Predicted image file {pred_file} does not exist.")
        check_predicted_values(pred_image)
        print('Check predicted values ok !')
    except ValueError as e:
        predicted_values_result["error"] = f"Error with predicted image content {pred_file}: {e}"
        print(f"Check predicted values failed !\nLook at {os.path.join(output_folder,'validation_results.json')} for more information !")

    # Validate CSV values
    try:
        if not os.path.exists(csv_file_path) or csv_file == 'None':
            raise ValueError(f"CSV file {csv_file} does not exist.")
        if not os.path.exists(pred_file_path) or pred_file == 'None' or pred_image == 'None':
            raise ValueError(f"Predicted image file {pred_file} does not exist.")
        check_csv_values(csv_file_path, pred_image)
        print('Check csv values ok !')
    except ValueError as e:
        csv_values_result["error"] = f"Error with CSV file content {csv_file}: {e}"
        print(f"Check csv values failed !\nLook at {os.path.join(output_folder,'validation_results.json')} for more information !")

    # Append results
    results["Image properties"].append(image_properties_result)
    results["Predicted image values"].append(predicted_values_result)
    results["CSV values"].append(csv_values_result)

    output_json_path = os.path.join(args.output_folder, "validation_results.json")
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Validation results saved to {output_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate T2 sagittal image outputs for consistency."
    )
    parser.add_argument('-i', '--input_folder', required=True, help="Path to the input folder containing pipelines inputs.")
    parser.add_argument('-o', '--output_folder', required=True, help="Path to the output folder containing pipelines outputs.")
    parser.add_argument('-debug', '--debug', action='store_true', help='Prints some information.')

    args = parser.parse_args()

    main(args)

    # Example command line to check data :
    # python3 data_check.py -i root_folder/ -o pred/