# Training and inference of the multi-modal model

This folder contains the files necessary for training, performing inference and evaluating the multi-modal model.

The model is a multimodal nnUNet so the training uses the built-in `nnUNetv2_train`function. 
To perform inference we also used the built-in `nnUNetv2_predict` function. 

More information on how to use these functions can be found on this repo. 
In particular the trainers we used are available in the branch `my_trainers`. 
For the challenge we used the DA5_150_epochs. 
For the challenge we used dataset 170 (cf preprocessing). 
