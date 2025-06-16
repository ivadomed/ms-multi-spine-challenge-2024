# Training and inference of the mono-modal model

This folder contains the files necessary for training, performing inference and evaluating the mono-modal model.

Scripts: 
- [build_json.py](build_json.py): This script is used to build a json file describing the entire dataset to facilitate inference and evaluation by storing all subject path in the dictionnary. 
- [images_dict.json](images_dict.json): the dictionnary built by the above script
- [run_evaluations.py](run_evaluations.py): This script runs the evaluation of the predictions. It uses the script evaluate_predictions.py in the evaluation branch.
- [run_inference_contrast_raw_space.py](run_inference_contrast_raw_space.py): This script runs the model inferenece on the images in the json dictionnary. It uses the model of which the path has been passed as an argument. It does the inference for models which segment lesions in the contrasts raw space (i.e. model 251 for instance)
- [run_inference_T2w_raw_space.py](run_inference_T2w_raw_space.py): This script runs the model inferenece on the images in the json dictionnary. It uses the model of which the path has been passed as an argument. It does the inference for models which segment lesion in the T2w raw space (i.e. model 151 for instance).

The [training_sc](training_script) folder contains the scripts for training the model. Training 100 corresponds to the final model (model 151). Training script 200 corresponds to model 251. Other trainings scripts are available in the project drive.