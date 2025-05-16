#!/bin/bash

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# activate environment
echo "Activating environment ..."
source /home/p/plb/links/projects/aip-jcohen/plb/challenge/.venv_nnunet/bin/activate        # TODO: update to match the name of your environment

# Define paths used:
PATH_NNUNET_RAW_FOLDER="/home/p/plb/links/projects/aip-jcohen/plb/challenge/nnUNet_raw"
PATH_OUTPUT="/home/p/plb/links/scratch/challenge"

# Create the nnUNet_preprocessed and nnUNet_results folders
mkdir -p $PATH_OUTPUT/nnUNet_preprocessed
mkdir -p $PATH_OUTPUT/nnUNet_results

# Export nnUNet paths
export nnUNet_raw=${PATH_NNUNET_RAW_FOLDER}
export nnUNet_preprocessed=${PATH_OUTPUT}/nnUNet_preprocessed
export nnUNet_results=${PATH_OUTPUT}/nnUNet_results

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"    
echo "nnUNet_results: $nnUNet_results"

# Define dataset values
cuda_device=3
dataset_number=150
configurations="3d_fullres"
fold=all
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetL_nonIso_Model1_Plans"
trainer="nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_500epochs"
model_checkpoint="checkpoint_final.pth"
pretrained_model_path="/home/p/plb/links/projects/aip-jcohen/plb/challenge/pretrained_multitalent/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres/fold_all/checkpoint_final.pth"


# Model training:
echo ""
echo "Training the model"
CUDA_VISIBLE_DEVICES=$cuda_device nnUNetv2_train  $dataset_number  $configurations $fold -p $plans -tr $trainer --c