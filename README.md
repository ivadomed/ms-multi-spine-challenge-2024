# ms-spine-challenge
Deep learning models for the MS-Multi-Spine challenge 2024: https://portal.fli-iam.irisa.fr/ms-multi-spine/

More information can be found in the Zenodo record: https://zenodo.org/records/14051168


# Files explanation 

The file `train.py`can be launched in order to train a model on the 


Run 
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"


nnUNetv2_plan_and_preprocess -d 30 -pl nnUNetPlannerResEncL --verify_dataset_integrity

nnUNetv2_train 30 3d_fullres 0 --npz -p nnUNetResEncUNetLPlans
