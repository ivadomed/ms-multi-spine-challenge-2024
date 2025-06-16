# Post-processing experiments

The scripts in this folder were designed to calibrate the post-processing of the model.

Scripts: 
- [1_prepare_experiment.py](1_prepare_experiment.py): In this script, we prepare the experiment (for monomodal 151) by creating folders for each subject with the necessary files for evaluation. This way, we aim at increase the speed of testing. 
- [2_run_inference.py](2_run_inference.py): In this script we run the inference (for model 151). It also averages the 5 fold predictions.
- [3_rmv_lesions_outside_sc.py](3_rmv_lesions_outside_sc.py): It removes lesions outside of the spinal cord. 
- [4_eval_rmv_lesion_outside_sc.py](4_eval_rmv_lesion_outside_sc.py): This script evaluates the performance increase of the post-processing step which removes the lesions outside the spinal cord.
- [5_rmv_lesion_max_value.py](5_rmv_lesion_max_value.py): This script removes lesions in predictions if the max value of the lesion mask is below a certain threshold. Thresholds investigated were between [0.5, 0.95] with a step size of 0.05.
- [6_eval_rmv_lesion_max_value.py](6_eval_rmv_lesion_max_value.py): This script evaluates the performance of removal of lesions based on their max voxel value. 
- [7_merge_predictions.py](7_merge_predictions.py): This script merges both prediction by doing a simple average of both predictions. A threshold of 0.8 was chosen for removal of lesions based on max value.
- [8_binarization_calibration.py](8_binarization_calibration.py): This script performs calibration of the threshold during mask binarization. The threshold investigated were between 0.3 to 0.95 in steps of 0.05.
- [9_eval_binarization_calibration.py](9_eval_binarization_calibration.py): This script evaluates the performance of the calibration of the model.
- [10_rmv_small_lesions.py](10_rmv_small_lesions.py): This script removes lesions in predictions depending on their sizes (in number of voxels). A 0.5 threshold is applied for binarization. Investigation was done on [5, 60] voxels with a step size of 5.
- [11_eval_rmv_small_lesions.py](11_eval_rmv_small_lesions.py): This script evaluates the performance of the calibration of the model.
- [12_convert_to_instance_seg.py](12_convert_to_instance_seg.py): This scripts computes the lesion_probbaility for each lesion instance. A threshold of 50 voxels was chosen for removal of small lesions.