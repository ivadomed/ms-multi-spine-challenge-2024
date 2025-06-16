# Evaluation

The scripts in this folder were designed so that we can perform evaluation of model's performance.

Scripts: 
- [evaluate_instance_pred.py](evaluate_instance_pred.py): This script evaluates instance segmentations performed by a model. 
- [evaluate_semantic_pred.py](evaluate_semantic_pred.py): This file is used to evaluate a semantic segmentation of the model.
- [evaluate_predictions.py](evaluate_predictions.py): This file is used to evaluate the predictions of the model on the test set. It is based on the format of the nnUnet storage of files.
- [plot_performance.py](plot_performance.py): This script is used to plot the performance of the model on the test set, validation and train set. It saves a plot of dice scores per contrat in the same folder as the text folder.
