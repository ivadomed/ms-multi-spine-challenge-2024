#!/bin/bash
set -e  

# ======================
# SCT installation
# ======================

#pip install wget 

#git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

#yes | ./spinalcordtoolbox/install_sct



# =====================
# CONFIGURATION BLOCK
# =====================
# Customize these for testing or submission

# Dataset directories
DATASET_NAME="MsMultiSpine"
DATASET_NUMBER="15"
DATASET_ID="150"
RAW_DATA="ms-multi-spine-challenge-2024"
NNUNET_DATASET="nnunet_datasets"
RESULTS="/path/to/output_results"

# nnU-Net trainer/plan
CONFIG="2d"
FOLD="0"
TRAINER="nnUNetTrainerDiceCELoss_noSmooth_300epochs"
PLAN="nnUNetResEncUNetLPlans"
DEVICE="cpu"  # or "cpu"

# other 
GITHUB="ms-multi-spine-challenge-2024-github"
# =====================



echo "Step 1: Preprocessing"

python $GITHUB/data_preproc/convert_to_nnunet_submission.py \
    --data $RAW_DATA \
    --output $NNUNET_DATASET \
    --task-name $DATASET_NAME \
    --task-number $DATASET_ID \
    --dataset-type $DATASET_NUMBER


echo "Step 2: Inference"

nnUNetv2_predict \
  -i $NNUNET_DATASET/Dataset$DATASET_ID_$DATASET_NAME/imagesTr \
  -o result \
  -d Dataset$DATASET_ID_$DATASET_NAME \
  -tr nnUNetTrainerDiceCELoss_noSmooth_150epochs \
  -c 2d \
  -p nnUNetResEncUNetLPlans \
  -f 0 \
  -device cpu \
  -npp 1 \
  -nps 1


