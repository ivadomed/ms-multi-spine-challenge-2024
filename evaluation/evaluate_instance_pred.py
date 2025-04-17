"""
This script evaluates instance segmentations performed by a model. 

Input:
    -pred: A segmentation with instance segmentations (each lesion is a separate instance)
    -pred_csv: A CSV file containing the prediction results (the probability of each instance)
    -gt: A segmentation with instance segmentations (each lesion is a separate instance)

Output:
    -results: A dictionary containing the evaluation results

Example usage: 
    python evaluate_instance_segmentation.py --pred pred.nii.gz --pred_csv pred.csv --gt gt.nii.gz

Author: Pierre-Louis Benveniste
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate instance segmentation.")
    parser.add_argument("--pred", type=str, required=True, help="Path to the predicted segmentation file.")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to the predicted CSV file.")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth segmentation file.")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    pred_path = args.pred
    pred_csv_path = args.pred_csv
    gt_path = args.gt

    


if __name__ == "__main__":
    main()