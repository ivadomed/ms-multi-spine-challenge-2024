"""
This script merges both prediction by doing a simple average of both predictions.
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

        # Build path to the lesion masks
        lesion_mask_t2 = os.path.join(sub_folder,"predictions", "t2w_segmentation_masked.nii.gz")
        lesion_mask_psir = os.path.join(sub_folder,"predictions", "psir_segmentation_masked.nii.gz")

        ## THE FOLLOWING NEEDS TO BE UPDATED (BECAUSE OF PROBLEM OF MASK FOV)

        # # Add both segmentation masks
        # assert os.system(f"sct_maths -i {lesion_mask_t2} -add {lesion_mask_psir} -o {os.path.join(sub_folder,'predictions', 'merged_segmentation_masked.nii.gz')}") == 0
        # ## Divide by 2 to average the predictions
        # assert os.system(f"sct_maths -i {os.path.join(sub_folder,'predictions', 'merged_segmentation_masked.nii.gz')} -div 2 -o {os.path.join(sub_folder,'predictions', 'merged_segmentation_masked.nii.gz')}") == 0
        # break


if __name__ == "__main__":
    main()
