"""
This script is used to run the inference on the data. It takes as input the path to nnUnet format datasets.

Input:
    -i: Path to the input data
    -o: Path to the output data
    -m: Path to the model

Output:
    None

Example:
    python run_inference.py -i /path/to/input/data -o /path/to/output/data -m /path/to/model

Author: Pierre-Louis Benveniste
"""

import os
import argparse
from pathlib import Path
import glob

os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor   # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from image import Image, get_orientation
import torch
import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on the data')
    parser.add_argument('-i', '--input', help='Path to the input data', required=True)
    parser.add_argument('-o', '--output', help='Path to the output data', required=True)
    parser.add_argument('-m', '--model', help='Path to the model', required=True)
    args = parser.parse_args()
    return args


def main():

    # Get arguments
    args = parse_args()
    data_path = args.input
    output_path = args.output
    model_path = args.model

    # Build the output folder
    os.makedirs(output_path, exist_ok=True)
    # We create a subfolder for each model fold
    output_paths_binary = []
    output_paths_soft = []
    for i in range(5):
        output_paths_binary.append(os.path.join(output_path, f'fold_{i}_binary'))
        os.makedirs(output_paths_binary[i], exist_ok=True)
        output_paths_soft.append(os.path.join(output_path, f'fold_{i}_soft'))
        os.makedirs(output_paths_soft[i], exist_ok=True)

    # Get all the images in the input folder (which is supposedly either imagesTs or imagesTr)
    images = list(Path(data_path).rglob('*.nii.gz'))

    # Iterat over the 5 folds of the model
    for fold in range(5):
        # For each image, run the inference
        # We initialize the model here
        tile_step_size = 0.5
        # get the nnunet trainer directory
        trainer_dirs = glob.glob(os.path.join(model_path, "nnUNetTrainer*"))
        trainer_path = trainer_dirs[0]
        folds_avail = [fold]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_name = 'checkpoint_final.pth'

        # instantiate the nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,  # changing it from 0.5 to 0.9 makes inference faster
            use_gaussian=True,  # applies gaussian noise and gaussian blur
            use_mirroring=True,  # test time augmentation by mirroring on all axes
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        print(f'Running inference on device: {predictor.device}')

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(trainer_path),
            use_folds=folds_avail,
            checkpoint_name=checkpoint_name,
        )
    
        # Iterate over the images
        for image in images:
            # Load the image
            img_in = Image(str(image))
            orig_orientation = get_orientation(img_in)
            model_orientation = 'RPI'
            if orig_orientation != model_orientation:
                img_in.change_orientation(model_orientation)

            data = img_in.data.transpose([2, 1, 0])

            data = np.expand_dims(data, axis=0).astype(np.float32)
            pred, prob = predictor.predict_single_npy_array(
                input_image=data,
                # The spacings also have to be reversed to match nnUNet's conventions.
                image_properties={'spacing': img_in.dim[6:3:-1]},
                save_or_return_probabilities=True
            )
            # Lastly, we undo the transpose to return the image from [z,y,x] (SimpleITK) to [x,y,z] (nibabel)
            pred = pred.transpose([2, 1, 0])
            img_out = img_in.copy()
            img_out.data = pred
            # Same for the prob
            prob = prob[1, ...]  # We only take the first channel (the 0th channel is the background)
            prob = prob.transpose([2, 1, 0])
            prob_out = img_in.copy()
            prob_out.data = prob

            if orig_orientation != model_orientation:
                img_out.change_orientation(orig_orientation)
                prob_out.change_orientation(orig_orientation)

            # Save the output segmentation
            output_file_path = os.path.join(output_paths_binary[fold], os.path.basename(image).replace('_0000.nii.gz', '.nii.gz'))
            print(f'Saving prediction to {output_file_path}')
            img_out.save(output_file_path)
            # Save the output probability
            output_prob_path = os.path.join(output_paths_soft[fold], os.path.basename(image).replace('_0000.nii.gz', '.nii.gz'))
            prob_out.save(output_prob_path)

    print('Inference done!')

    return None


if __name__ == '__main__':
    main()