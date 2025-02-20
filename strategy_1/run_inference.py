"""
This script is used to run the inference on the data.

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
    path_natural = os.path.join(output_path, 'naturalImages')
    os.makedirs(path_natural, exist_ok=True)
    path_preprocessed = os.path.join(output_path, 'preprocessedImages')
    os.makedirs(path_preprocessed, exist_ok=True)
    path_registered = os.path.join(output_path, 'registeredImages')
    os.makedirs(path_registered, exist_ok=True)

    output_paths = [path_natural, path_preprocessed, path_registered]

    # Get all the images
    ## Get natural images with rglob
    natural_images = list(Path(data_path).rglob('*.nii.gz'))
    natural_images = [x for x in natural_images if 'derivatives' not in str(x)]
    natural_images = [x for x in natural_images if 'preproc' not in str(x)]
    natural_images = [x for x in natural_images if 'SHA256E' not in str(x)]
    ## Get all preprocessed images
    preprocessed_images = list(Path(data_path).rglob('*preproc*.nii.gz'))
    preprocessed_images = [x for x in preprocessed_images if 'derivatives' not in str(x)]
    preprocessed_images = [x for x in preprocessed_images if 'preprocReg' in str(x)]
    preprocessed_images = [x for x in preprocessed_images if 'SHA256E' not in str(x)]
    ## Get all registered images
    registered_images = list(Path(data_path).rglob('*preprocReg*.nii.gz'))
    registered_images = [x for x in registered_images if 'derivatives' not in str(x)]
    registered_images = [x for x in registered_images if 'SHA256E' not in str(x)]

    # Itertate over the images of the 3 folders
    for i, folder in enumerate([natural_images, preprocessed_images, registered_images]):
        for image in folder:
            output_preds = []
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
                    use_mirroring=False,  # test time augmentation by mirroring on all axes
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

                # Create a temp path for the image
                tmpdir = os.path.join(output_path, 'tmp')
                os.makedirs(tmpdir, exist_ok=True)
                
                img_in = Image(str(image))
                orig_orientation = get_orientation(img_in)
                model_orientation = 'RPI'
                if orig_orientation != model_orientation:
                    img_in.change_orientation(model_orientation)
                
                # Create directory for nnUNet prediction
                tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')

                data = img_in.data.transpose([2, 1, 0])

                data = np.expand_dims(data, axis=0).astype(np.float32)
                pred = predictor.predict_single_npy_array(
                    input_image=data,
                    # The spacings also have to be reversed to match nnUNet's conventions.
                    image_properties={'spacing': img_in.dim[6:3:-1]},
                )
                # Lastly, we undo the transpose to return the image from [z,y,x] (SimpleITK) to [x,y,z] (nibabel)
                pred = pred.transpose([2, 1, 0])
                img_out = img_in.copy()
                img_out.data = pred

                if orig_orientation != model_orientation:
                    img_out.change_orientation(orig_orientation)

                # Save the output segmentation
                output_file_path = os.path.join(output_paths[i], os.path.basename(image).replace('.nii.gz', f'_pred_fold{fold}.nii.gz'))
                print(f'Saving prediction to {output_file_path}')
                img_out.save(output_file_path)

                # Save the output image
                output_preds.append(output_file_path)

                # Clean up
                os.rmdir(tmpdir)

            # Using the 5 predictions, we create an avg prediction
            seg_avg = nib.load(output_preds[0])
            seg_avg_data = seg_avg.get_fdata()
            for k in range(1,5):
                seg_avg_data += nib.load(output_preds[k]).get_fdata()  
            seg_avg_data /= 5
            img_avg = nib.Nifti1Image(seg_avg_data, seg_avg.affine, seg_avg.header)
            nib.save(img_avg,os.path.join(output_paths[i], os.path.basename(image).replace('.nii.gz', '_pred_avg.nii.gz')))

            # Save a binary version of the average prediction
            seg_avg_data[seg_avg_data < 0.5] = 0
            seg_avg_data[seg_avg_data >= 0.5] = 1
            img_avg = nib.Nifti1Image(seg_avg_data, seg_avg.affine, seg_avg.header)
            nib.save(img_avg,os.path.join(output_paths[i], os.path.basename(image).replace('.nii.gz', '_pred_avg_bin.nii.gz')))

    print('Inference done!')

    return None


if __name__ == '__main__':
    main()