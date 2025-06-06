"""
This script performs post-processing on the results of experiment 251.
It removes lesions outside of the spinal cord. 
"""
import json
import os

def main():

    image_dict = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/images_dict.json"

    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/exp_251_prep"

    # load the json file
    with open(image_dict, "r") as f:
        images_dict = json.load(f)

    training_images = images_dict["training"]
    testing_images = images_dict["testing"]
    images = {**training_images, **testing_images}

    # iterate over the images
    for image in images:
        # If contrast is T2, we skip
        if images[image]["contrast"] == "T2w":
            continue
        # Build a folder for each subject
        sub_folder = os.path.join(output_folder, images[image]["subject_name"])
        print("Subject name", images[image]["subject_name"])

        # Build path to sc seg of the t2w raw image
        sc_seg_t2_raw = os.path.join(sub_folder, "sc_seg_t2_raw.nii.gz")

        # Build path to the lesion masks
        lesion_mask_t2 = os.path.join(sub_folder,"predictions", "t2w_segmentation.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder,"predictions", "psir_segmentation.nii.gz")

        # We dilate the SC mask by 1 voxel in all directions
        assert os.system(f"sct_maths -i {sc_seg_t2_raw} -dilate 1 -o {sub_folder}/sc_seg_t2_raw_dilated.nii.gz")==0

        # We remove the lesions outside of the spinal cord
        assert os.system(f"sct_maths -i {lesion_mask_t2} -mul {sub_folder}/sc_seg_t2_raw_dilated.nii.gz -o {lesion_mask_t2.replace('.nii.gz', '_masked.nii.gz')}")==0
        assert os.system(f"sct_maths -i {lesion_mask_psir} -mul {sub_folder}/sc_seg_t2_raw_dilated.nii.gz -o {lesion_mask_psir.replace('.nii.gz', '_masked.nii.gz')}")==0

        break

if __name__ == "__main__":
    main()