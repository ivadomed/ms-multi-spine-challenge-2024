import os
import argparse
import pandas as pd
import numpy as np


def load_data(gt_path, pred_path):
    tableGT = pd.read_csv(gt_path, sep='\t')
    tablePred = pd.read_csv(pred_path, sep='\t')
    return tableGT, tablePred


def filter_by_proba_threshold(tableGT, tablePred, proba_threshold):
    # Create copies of the input DataFrames
    filtered_tableGT = tableGT.copy()
    filtered_tablePred = tablePred.copy()

    # Create masks for rows that meet the probability threshold
    maskGT = filtered_tableGT['predicted_proba'] >= proba_threshold
    maskPred = filtered_tablePred['predicted_proba'] >= proba_threshold

    # Replace rows in the copied DataFrames that do not meet the threshold
    filtered_tableGT.loc[~maskGT, ['predicted_instance_id', 'IoU', 'predicted_proba', 'is_true_positive']] = None
    filtered_tablePred.loc[~maskPred, ['predicted_instance_id', 'is_false_positive', 'predicted_proba']] = None

    return filtered_tableGT, filtered_tablePred


def compute_image_level_metrics(args, filtered_tableGT, filtered_tablePred, iou_threshold=0.2):
    image_level_data = []

    for image_name, gt_group in filtered_tableGT.groupby('image_name'):
        pred_group = filtered_tablePred[filtered_tablePred['image_name'] == image_name]

        # Filter gt_group to only include rows where the IoU is above the threshold
        matched_references = gt_group[gt_group['IoU'] >= iou_threshold]['reference_instance_id'].unique()
        number_of_true_positives = len(matched_references)  # Count each reference_instance_id only once
        
        # Count false positives from predictions that did not match any reference_instance_id
        number_of_false_positives = pred_group['is_false_positive'].sum()
        number_of_reference_lesions = gt_group['reference_instance_id'].nunique()
        number_of_predicted_lesions = pred_group['predicted_instance_id'].nunique()

        # Compute image-level sensitivity, precision, and F1 score
        sensitivity = (number_of_true_positives / number_of_reference_lesions) if number_of_reference_lesions > 0 else np.nan
        precision = (number_of_true_positives / (number_of_true_positives + number_of_false_positives)) if (number_of_true_positives + number_of_false_positives) > 0 else 1
        if precision == 0 and sensitivity == 0:
            f1_score = 0
        elif np.isnan(sensitivity):
            # No reference lesions, so precision is either 0 (no predictions) or 1 (at least one prediction)
            f1_score = precision
        else:
            f1_score = (2 * precision * sensitivity / (precision + sensitivity))

        image_level_data.append({
            'image_name': image_name,
            'number_of_true_positives': number_of_true_positives,
            'number_of_false_positives': number_of_false_positives,
            'number_of_reference_lesions': number_of_reference_lesions,
            'number_of_predicted_lesions': number_of_predicted_lesions,
            'sensitivity': sensitivity,
            'precision': precision,
            'F1_score': f1_score
        })
    
    # Create a DataFrame for image-level metrics
    image_level_df = pd.DataFrame(image_level_data)
    return image_level_df


def calculate_average_metrics(image_level_df):
    # Calculate the average of image-level sensitivity, precision, and F1 scores
    mean_sensitivity = image_level_df['sensitivity'].mean()
    mean_precision = image_level_df['precision'].mean()
    mean_f1_score = image_level_df['F1_score'].mean()
    mean_fp_per_image = image_level_df['number_of_false_positives'].mean()
    return mean_sensitivity, mean_precision, mean_f1_score, mean_fp_per_image


def trapezoidal_interpolation(lower_value, upper_value, lower_metric, upper_metric, target_metric):
    # Linear interpolation for a single metric
    if upper_metric != lower_metric:
        return lower_value + (target_metric - lower_metric) * (upper_value - lower_value) / (upper_metric - lower_metric)
    return lower_value


def filter_table_and_get_metrics(args, tableGT, tablePred, threshold, i=0, iou_threshold=0.2):
    filtered_tableGT, filtered_tablePred = filter_by_proba_threshold(tableGT, tablePred, threshold)
    image_level_df = compute_image_level_metrics(args, filtered_tableGT, filtered_tablePred, iou_threshold)

    if args.debug:
        image_tsv_path = os.path.join(args.output_path, f'image_level_{threshold}_iteration_{i}.tsv')
        image_level_df.to_csv(image_tsv_path, sep='\t', index=False)

    return calculate_average_metrics(image_level_df)


def binary_search_for_threshold(args, target_metric, tableGT, tablePred, metric_type, iou_threshold=0.2):
    # Extract sorted unique probability values from predicted probabilities
    unique_probas = sorted(tablePred['predicted_proba'].unique())
    unique_probas.insert(0,0)
    unique_probas.append(1)
    unique_probas.append(1)
    low, high = 0, len(unique_probas) - 1

    lower_metric_info = None
    upper_metric_info = None

    # Perform binary search 
    i=0
    while low < high-1:
        i += 1
        mid = (low + high) // 2
        proba_threshold = unique_probas[mid]

        # Filter tables by current threshold, and calculate image-level and average metrics
        mean_sensitivity, mean_precision, mean_f1_score, mean_fp_per_image = \
            filter_table_and_get_metrics(args, tableGT, tablePred, proba_threshold, i, iou_threshold)

        # Select the metric of interest
        current_metric = {
            'mean_fp_per_image': mean_fp_per_image,
            'mean_sensitivity': mean_sensitivity,
            'mean_precision': mean_precision,
            'mean_f1_score': mean_f1_score
        }[metric_type]

        if current_metric == target_metric:
            # equality with target metric, stopping iterations
            while current_metric == target_metric:
                proba_threshold_previous, mean_sensitivity_previous, mean_precision_previous, mean_f1_score_previous, mean_fp_per_image_previous = \
                    proba_threshold, mean_sensitivity, mean_precision, mean_f1_score, mean_fp_per_image
                mid -= 1
                proba_threshold = unique_probas[mid]
                mean_sensitivity, mean_precision, mean_f1_score, mean_fp_per_image = \
                    filter_table_and_get_metrics(args, tableGT, tablePred, proba_threshold, i, iou_threshold)
                current_metric = {
                    'mean_fp_per_image': mean_fp_per_image,
                    'mean_sensitivity': mean_sensitivity,
                    'mean_precision': mean_precision,
                    'mean_f1_score': mean_f1_score
                }[metric_type]

            return proba_threshold_previous, mean_sensitivity_previous, mean_precision_previous, mean_f1_score_previous

        elif current_metric > target_metric:
            low = mid
            lower_metric_info = (low, proba_threshold, mean_sensitivity, mean_precision, mean_f1_score, current_metric)

        else:
            high = mid
            upper_metric_info = (high, proba_threshold, mean_sensitivity, mean_precision, mean_f1_score, current_metric)

    try:
        _, lower_threshold, lower_sensitivity, lower_precision, lower_f1, lower_metric = lower_metric_info
        _, upper_threshold, upper_sensitivity, upper_precision, upper_f1, upper_metric = upper_metric_info
        best_threshold = trapezoidal_interpolation(lower_threshold, upper_threshold, lower_metric, upper_metric, target_metric)
        best_mean_sensitivity = trapezoidal_interpolation(lower_sensitivity, upper_sensitivity, lower_threshold, upper_threshold, best_threshold)
        best_mean_precision = trapezoidal_interpolation(lower_precision, upper_precision, lower_threshold, upper_threshold, best_threshold)
        best_mean_f1_score = trapezoidal_interpolation(lower_f1, upper_f1, lower_threshold, upper_threshold, best_threshold)
        return best_threshold, best_mean_sensitivity, best_mean_precision, best_mean_f1_score
    except:
        return proba_threshold, mean_sensitivity, mean_precision, mean_f1_score 


def single_binary_search(args, gt_path, pred_path, target_metric, metric_type, output_path, iou_threshold=0.2):
    # Load data
    tableGT, tablePred = load_data(gt_path, pred_path)

    # Binary search for optimal threshold based on target metric
    final_threshold, mean_sensitivity, mean_precision, mean_f1_score = binary_search_for_threshold(
        args, target_metric, tableGT, tablePred, metric_type, iou_threshold
    )

    if args.debug:
        # Save the final image-level metrics to TSV
        filtered_tableGT_final, filtered_tablePred_final = filter_by_proba_threshold(tableGT, tablePred, final_threshold)
        image_level_df_final = compute_image_level_metrics(args, filtered_tableGT_final, filtered_tablePred_final,
                                                           iou_threshold)
        image_tsv_path = os.path.join(output_path, f'image_level_{metric_type}_{target_metric}_{final_threshold}.tsv')
        image_level_df_final.to_csv(image_tsv_path, sep='\t', index=False)

    return final_threshold, mean_sensitivity, mean_precision, mean_f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Code dedicated to the generation of image level metrics, used to perform a binary search of the best
        threshold to obtain a given target metric value.""")
    
    parser.add_argument('-gt', '--gt_tsv_path', required=False, default='/dataEvalChallenge/tmp/table_gt.tsv', help='Path to the reference masks.')
    parser.add_argument('-pt', '--pred_tsv_path', required=False, default='/dataEvalChallenge/tmp/table_pred.tsv', help='Path to the predicted 3D nifti files.')
    parser.add_argument('-m', '--metric_type', required=False, type=str, default='mean_fp_per_image', help='Metrics to reach.')
    parser.add_argument('-t', '--target_metric', required=False, type=float, default=1.5, help='Value to reach for the selected target metrics.')
    parser.add_argument('-o', '--output_path', required=False, default='/dataEvalChallenge/tmp', help='Path to the output folder.')
    parser.add_argument('-debug', '--debug', action='store_true', help='Printing some intermediate information.')
    args = parser.parse_args()

    single_binary_search(args, args.gt_path, args.pred_path, args.target_metric, args.metric_type, args.output_path)
