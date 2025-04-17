import os
import argparse
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage import label


def load_nifti_file(file_path):
    return nib.load(file_path).get_fdata()


def compute_connected_components(nifti_data):
    # Label connected components in a 3D volume
    labeled_data, num_features = label(nifti_data, structure=np.ones((3,3,3)))
    return labeled_data, num_features


def compute_iou(gt_component, pred_component):
    # Compute IoU: intersection / union
    intersection = np.logical_and(gt_component, pred_component).sum()
    union = np.logical_or(gt_component, pred_component).sum()
    return intersection / union if union != 0 else 0


def generate_matching_tables(args, gt_folder, pred_folder, csv_folder, iou_threshold, image_list):
    # Initialize the final dataframes for all images
    all_gt_rows = []
    all_pred_rows = []

    for image in sorted(image_list):
        image_name = image.split('.nii')[0]
        # Define file paths for GT and prediction
        gt_path = os.path.join(gt_folder, image)
        pred_path = os.path.join(pred_folder, image)
        csv_path = os.path.join(csv_folder, f"{image_name}.csv")

        # Load GT, prediction, and CSV data
        gt_data = load_nifti_file(gt_path)
        pred_data = load_nifti_file(pred_path)
        pred_prob_df = pd.read_csv(csv_path)

        # Initialize rows for this image
        gt_rows = []
        pred_rows = []

        # Determine unique GT and prediction labels
        gt_labels = np.unique(gt_data[gt_data > 0])
        pred_labels = np.unique(pred_data[pred_data > 0])

        # Initialize TP/FP tracking dictionaries
        is_false_positive = {pred_id: 1 for pred_id in pred_labels}  # initially all are FP

        if gt_labels.size == 0:
            gt_rows.append({
                'image_name': image_name,
                'reference_instance_id': None,
                'predicted_instance_id': None,
                'IoU': None,
                'predicted_proba': None,
                'is_true_positive': 0
            })

        # Compute IoU for each GT-Pred pair
        for gt_id in gt_labels:
            gt_component = (gt_data == gt_id)
            max_iou = 0

            for pred_id in pred_labels:
                pred_component = (pred_data == pred_id)
                iou = compute_iou(gt_component, pred_component)
                
                # Find probability for this predicted component
                pred_proba = pred_prob_df.loc[pred_prob_df['label'] == pred_id, 'p'].values[0]

                # Append entry for tableGT
                if iou > 0:
                    gt_rows.append({
                        'image_name': image_name,
                        'reference_instance_id': gt_id,
                        'predicted_instance_id': pred_id,
                        'IoU': iou,
                        'predicted_proba': pred_proba,
                        'is_true_positive': int(iou > iou_threshold)
                    })

                    # Update max IoU if this IoU is higher
                    max_iou = max(max_iou, iou)

                    # If IoU is above threshold, mark the predicted instance as not FP
                    if iou >= iou_threshold:
                        is_false_positive[pred_id] = 0

            # Append row for GT component without any match above threshold
            if max_iou == 0:
                gt_rows.append({
                    'image_name': image_name,
                    'reference_instance_id': gt_id,
                    'predicted_instance_id': None,
                    'IoU': None,
                    'predicted_proba': None,
                    'is_true_positive': 0
                })

        # Populate tablePred based on FP analysis
        for pred_id, fp_flag in is_false_positive.items():
            pred_proba = pred_prob_df.loc[pred_prob_df['label'] == pred_id, 'p'].values[0]
            pred_rows.append({
                'image_name': image_name,
                'predicted_instance_id': pred_id,
                'is_false_positive': fp_flag,
                'predicted_proba': pred_proba
            })

        # Append current image's rows to all images' list
        all_gt_rows.extend(gt_rows)
        all_pred_rows.extend(pred_rows)

    # Create final DataFrames with all images' data
    tableGT = pd.DataFrame(all_gt_rows)
    tablePred = pd.DataFrame(all_pred_rows)

    return tableGT, tablePred


def generate_lesion_level_files(args, gt_path, pred_path, csv_path, iou_tresh, output_path):
    # Collect all .nii files from the gt_path as image list
    image_list = [im for im in os.listdir(gt_path) if '.nii' in im]
    tableGT, tablePred = generate_matching_tables(args, gt_path, pred_path, csv_path, iou_tresh, image_list)
    
    # Save the DataFrames as TSV files
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    tableGT.to_csv(os.path.join(output_path, 'table_gt.tsv'), sep='\t', index=False)
    tablePred.to_csv(os.path.join(output_path, 'table_pred.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Evaluation of predictions provided under the following form: a 3D nifti file in which the connected 
        components are associated with a unique value, as well as a csv file that provides the probabilities associated with each 
        predicted component for each image.""")
    
    parser.add_argument('-g', '--gt_path', required=False, default='/dataEvalChallenge/gt', help='Path to the reference masks.')
    parser.add_argument('-p', '--pred_path', required=False, default='/dataEvalChallenge/data', help='Path to the predicted 3D nifti files.')
    parser.add_argument('-c', '--csv_path', required=False, default='/dataEvalChallenge/data', help='Path to the csvs associated with the predicted niftis.')
    parser.add_argument('-t', '--iou_tresh', required=False, type=float, default=0.2, help='IoU threshold.')
    parser.add_argument('-o', '--output_path', required=False, default='/dataEvalChallenge/tmp', help='Path to the output folder.')
    args = parser.parse_args()

    generate_lesion_level_files(args, args.gt_path, args.pred_path, args.csv_path, args.iou_tresh, args.output_path)
