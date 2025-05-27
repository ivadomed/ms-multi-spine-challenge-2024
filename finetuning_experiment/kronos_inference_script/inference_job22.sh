#!/bin/bash

# activate environment
echo "Activating environment ..."
source activate venv_my_nnunet_challenge

# Export nnUNet paths
export nnUNet_raw="/home/plbenveniste/net/challenge-multi-spine/nnUNet_experiments/nnUNet_raw"
export nnUNet_preprocessed="/home/plbenveniste/net/challenge-multi-spine/nnUNet_experiments/nnUNet_preprocessed"
export nnUNet_results="/home/plbenveniste/net/challenge-multi-spine/compute_canada_results/nnUNet_results"

# Image to run inference on: 
path_image="/home/plbenveniste/net/challenge-multi-spine/nnUNet_experiments/nnUNet_raw/Dataset150_MsMultiSpine"
path_results="/home/plbenveniste/net/challenge-multi-spine/compute_canada_results/nnUNet_results"

# Define dataset values
cuda_device=3
dataset_number=140
configurations="3d_fullres"
fold=all
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetL_nonIso_Model1_Plans"
trainer="nnUNetTrainerDAExt_DiceCELoss_noSmooth_500epochs"

# Output_folder 
output_folder=Dataset${dataset_number}_MsMultiSpine/${trainer}__${plans}__${configurations}

# Inference command
CUDA_VISIBLE_DEVICES=$cuda_device nnUNetv2_predict -i $path_image/imagesTs -o $path_results/$output_folder/preds_foldall_chkBest_Ts -d $dataset_number -c $configurations -f $fold -chk checkpoint_best.pth -p $plans -tr $trainer

# Conda deactivate
echo "Deactivating environment ..."
conda deactivate

# Conda activate environment for evaluation
echo "Activating environment for evaluation ..."
source activate venv_eval_challenge

# Evaluation command
python ~/net/challenge-multi-spine/evaluation_scripts/ms-multi-spine-challenge-2024/evaluation/evaluate_predictions.py -pred-folder $path_results/$output_folder/preds_foldall_chkBest_Ts -label-folder $path_image/labelsTs/ -image-folder $path_image/imagesTs/ -conversion-dict $path_image/conversion_dict.json -output-folder $path_results/$output_folder/preds_foldall_chkBest_Ts

# Plot command
python ~/net/challenge-multi-spine/evaluation_scripts/ms-multi-spine-challenge-2024/evaluation/plot_performance.py --pred-dir-path $path_results/$output_folder/preds_foldall_chkBest_Ts

# Conda deactivate
echo "Deactivating environment ..."
conda deactivate

##########
########## Inference for training set
##########

# activate environment
echo "Activating environment ..."
source activate venv_my_nnunet_challenge

# Inference command
CUDA_VISIBLE_DEVICES=$cuda_device nnUNetv2_predict -i $path_image/imagesTr -o $path_results/$output_folder/preds_foldall_chkBest_Tr -d $dataset_number -c $configurations -f $fold -chk checkpoint_best.pth -p $plans -tr $trainer

# Conda deactivate
echo "Deactivating environment ..."
conda deactivate

# Conda activate environment for evaluation
echo "Activating environment for evaluation ..."
source activate venv_eval_challenge

# Evaluation command
python ~/net/challenge-multi-spine/evaluation_scripts/ms-multi-spine-challenge-2024/evaluation/evaluate_predictions.py -pred-folder $path_results/$output_folder/preds_foldall_chkBest_Tr -label-folder $path_image/labelsTr/ -image-folder $path_image/imagesTr/ -conversion-dict $path_image/conversion_dict.json -output-folder $path_results/$output_folder/preds_foldall_chkBest_Tr

# Plot command
python ~/net/challenge-multi-spine/evaluation_scripts/ms-multi-spine-challenge-2024/evaluation/plot_performance.py --pred-dir-path $path_results/$output_folder/preds_foldall_chkBest_Tr