import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from challenge_functions.binary_search import single_binary_search, filter_table_and_get_metrics, load_data


def calculate_froc(args, gt_path, pred_path, target_metric_list, metric_type, output_path, iou_threshold=0.2):
    mean_sensitivities = []
    results = {}

    # Run binary search for each target metric and collect sensitivities
    for target_metric in target_metric_list:
        final_threshold, mean_sensitivity, mean_precision, mean_f1_score = single_binary_search(
            args, gt_path, pred_path, target_metric, metric_type, output_path, iou_threshold
        )
        mean_sensitivities.append(mean_sensitivity)
        result_tmp = {
            f'{metric_type}_{target_metric}': {
                'threshold': final_threshold,
                'sensitivity': mean_sensitivity,
                'precision': mean_precision,
                'F1 score': mean_f1_score
            }
        }
        results.update(result_tmp)

    # Calculate the FROC metric as the average of the mean sensitivities
    froc_metric = np.mean(mean_sensitivities)

    # Calculate performance at threshold of 0.5
    tableGT, tablePred = load_data(gt_path, pred_path)
    mean_sensitivity, mean_precision, mean_f1_score, fp_per_image = filter_table_and_get_metrics(
        args, tableGT, tablePred, threshold=0.5, iou_threshold=iou_threshold
    )
    results['thresh_0.5'] = {
        'threshold': 0.5,
        'sensitivity': mean_sensitivity,
        'precision': mean_precision,
        'F1 score': mean_f1_score,
        'fp_per_image': fp_per_image,
    }

    # Writing results into a JSON file
    froc_dict = {"FROC Metric": froc_metric}
    results.update(froc_dict)
    results_json_path = os.path.join(output_path, 'results.json')
    with open(results_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Plot mean sensitivity vs target metrics
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=target_metric_list, y=mean_sensitivities, marker="o")
    plt.xlabel(metric_type)
    plt.ylabel("Mean Sensitivity")
    plt.title(f"Mean Sensitivity vs {metric_type}")
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_path, f"sensitivity_vs_{metric_type}.png")
    plt.savefig(plot_path)
    plt.close()

    if not args.debug:
        os.remove(os.path.join(output_path, 'table_gt.tsv'))
        os.remove(os.path.join(output_path, 'table_pred.tsv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Code calculating the FROC metric and generating a plot of the mean sensitivity as a function
        of the target metric.""")
    
    parser.add_argument('-gt', '--gt_tsv_path', required=False, default='/dataEvalChallenge/tmp/table_gt.tsv', help='Path to the reference masks.')
    parser.add_argument('-pt', '--pred_tsv_path', required=False, default='/dataEvalChallenge/tmp/table_pred.tsv', help='Path to the predicted 3D nifti files.')
    parser.add_argument('-m', '--metric_type', required=False, type=str, default='mean_fp_per_image', help='Metrics to reach.')
    parser.add_argument('-tm', '--target_metric_list', required=False, type=list, default=[0.25, 0.5, 1, 2, 3], help='Value to reach for the selected target metrics.')
    parser.add_argument('-o', '--output_path', required=False, default='/dataEvalChallenge/tmp', help='Path to the output folder.')

    parser.add_argument('-debug', '--debug', action='store_true', help='Printing some intermediate information.')
    args = parser.parse_args()

    calculate_froc(args, args.gt_path, args.pred_path, args.target_metric_list, args.metric_type, args.output_path)
