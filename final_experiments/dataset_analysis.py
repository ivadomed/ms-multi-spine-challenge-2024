"""
This script performs dataset analysis for the MS Multi-Spine Challenge 2024 dataset.

Author: Pierre-Louis Benveniste
"""

import os
from pathlib import Path
import pandas as pd
import json
from image import Image, get_dimension, change_orientation
import argparse
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset analysis for MS Multi-Spine Challenge 2024.")
    parser.add_argument("--path_dataset", type=str, required=True, help="Path to the dataset.")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    path_dataset = args.path_dataset

    # All images
    all_images = list(Path(path_dataset).rglob("*.nii.gz"))
    all_images = [str(image) for image in all_images if 'desc-preproc' not in str(image)]
    # Remove labels
    all_images = [image for image in all_images if 'derivatives' not in image]
    # Remove images with SHA256
    all_images = [image for image in all_images if 'SHA256' not in image]
    print(len(all_images), "images found in the dataset.")
    all_images

    # Build a dataset with the image name
    images_df = pd.DataFrame(all_images, columns=['image_path'])
    images_df['image_name'] = images_df['image_path'].apply(lambda x: os.path.basename(x))
    # Build an empty column manufacturer
    images_df['manufacturer'] = ''
    images_df['MRI_power'] = ''
    # Iterate over the images and get the manufacturer and MRI power
    for index, row in images_df.iterrows():
        # Get the image path
        image_path = row['image_path']
        # Get the JSON file associated with the image
        json_file = image_path.replace('.nii.gz', '.json')
        # Open the JSON file and get the manufacturer and MRI power
        with open(json_file, 'r') as f:
            json_data = f.read()
            json_data = json.loads(json_data)
            # Get the manufacturer and MRI power
            images_df.at[index, 'manufacturer'] = json_data.get('Manufacturer', '')
            images_df.at[index, 'MRI_power'] = json_data.get('MagneticFieldStrength', '')
    # Add a contrast column
    images_df['contrast'] = images_df['image_name'].apply(lambda x: x.split('_')[-1].replace('.nii.gz', ''))

    # For each image, we want to add the resolution (making sure the image is in RPI)
    images_df['R-L_resolution'] = ''
    images_df['A-P_resolution'] = ''
    images_df['I-S_resolution'] = ''
    for index, row in images_df.iterrows():
        image_path = row['image_path']
        image_loaded = Image(image_path)
        # Set orientation to RPI
        image_reoriented = image_loaded.change_orientation('RPI')
        # Get the resolution
        resolution = get_dimension(image_reoriented)
        # Add the resolution to the dataframe
        images_df.at[index, 'R-L_resolution'] = resolution[4]
        images_df.at[index, 'A-P_resolution'] = resolution[5]
        images_df.at[index, 'I-S_resolution'] = resolution[6]

    # Output folder : 
    output_folder = "/home/plbenveniste/net/challenge-multi-spine/final_compute_canada_results/data_analysis_output"

    # Save the dataset to a csv file

    # Create logs to save in the output folder
    # Build a table showing the distribution of manufacturer
    table_manufacturer = PrettyTable()
    table_manufacturer.field_names = ['Manufacturer', 'Count']
    for manufacturer, group in images_df.groupby('manufacturer'):
        table_manufacturer.add_row([manufacturer, len(group)])
    print(table_manufacturer)

    # Build a table showing the distribution of MRI field strength
    table_mri_power = PrettyTable()
    table_mri_power.field_names = ['MRI Power', 'Count']
    for mri_power, group in images_df.groupby('MRI_power'):
        table_mri_power.add_row([mri_power, len(group)])
    print(table_mri_power)

    # Build a table showing the distribution of contrast
    table_contrast = PrettyTable()
    table_contrast.field_names = ['Contrast', 'Count']
    for contrast, group in images_df.groupby('contrast'):
        table_contrast.add_row([contrast, len(group)])
    print(table_contrast)

    # Build a table showing the distribution of resolution
    table_resolution = PrettyTable()
    table_resolution.field_names = ['', 'R-L Resolution', 'A-P Resolution', 'I-S Resolution']
    ## In this case we want max, mean, median, min and std of the resolution
    table_resolution.add_row(['Mean', round(images_df['R-L_resolution'].mean(), 2),
                                round(images_df['A-P_resolution'].mean(), 2),
                                round(images_df['I-S_resolution'].mean(), 2)])
    table_resolution.add_row(['Max', images_df['R-L_resolution'].max(),
                                images_df['A-P_resolution'].max(),
                                images_df['I-S_resolution'].max()])
    table_resolution.add_row(['Min', images_df['R-L_resolution'].min(),
                                images_df['A-P_resolution'].min(),
                                images_df['I-S_resolution'].min()])
    table_resolution.add_row(['Median', images_df['R-L_resolution'].median(),
                                images_df['A-P_resolution'].median(),
                                images_df['I-S_resolution'].median()])
    table_resolution.add_row(['Std', round(images_df['R-L_resolution'].std(), 2),
                                round(images_df['A-P_resolution'].std(), 2),
                                round(images_df['I-S_resolution'].std(), 2)])
    print(table_resolution)

    # Filter only T2w images
    t2w_images_df = images_df[images_df['contrast'] == 'T2w']
    # Build a table showing the distribution of resolution
    table_resolution_t2 = PrettyTable()
    table_resolution_t2.field_names = ['', 'R-L Resolution', 'A-P Resolution', 'I-S Resolution']
    ## In this case we want max, mean, median, min and std of the resolution
    table_resolution_t2.add_row(['Mean', round(t2w_images_df['R-L_resolution'].mean(), 2),
                                round(t2w_images_df['A-P_resolution'].mean(), 2),
                                round(t2w_images_df['I-S_resolution'].mean(), 2)])
    table_resolution_t2.add_row(['Max', t2w_images_df['R-L_resolution'].max(),
                                t2w_images_df['A-P_resolution'].max(),
                                t2w_images_df['I-S_resolution'].max()])
    table_resolution_t2.add_row(['Min', t2w_images_df['R-L_resolution'].min(),
                                t2w_images_df['A-P_resolution'].min(),
                                t2w_images_df['I-S_resolution'].min()])
    table_resolution_t2.add_row(['Median', t2w_images_df['R-L_resolution'].median(),
                                t2w_images_df['A-P_resolution'].median(),
                                t2w_images_df['I-S_resolution'].median()])
    table_resolution_t2.add_row(['Std', round(t2w_images_df['R-L_resolution'].std(), 2),
                                round(t2w_images_df['A-P_resolution'].std(), 2),
                                round(t2w_images_df['I-S_resolution'].std(), 2)])
    print("T2w Resolution Statistics:")
    print(table_resolution_t2)

    # Filter only PSIR images
    psir_images_df = images_df[images_df['contrast'] == 'PSIR']
    # Build a table showing the distribution of resolution
    table_resolution_psir = PrettyTable()
    table_resolution_psir.field_names = ['', 'R-L Resolution', 'A-P Resolution', 'I-S Resolution']
    ## In this case we want max, mean, median, min and std of the resolution
    table_resolution_psir.add_row(['Mean', round(psir_images_df['R-L_resolution'].mean(), 2),
                                round(psir_images_df['A-P_resolution'].mean(), 2),
                                round(psir_images_df['I-S_resolution'].mean(), 2)])
    table_resolution_psir.add_row(['Max', psir_images_df['R-L_resolution'].max(),
                                psir_images_df['A-P_resolution'].max(),
                                psir_images_df['I-S_resolution'].max()])
    table_resolution_psir.add_row(['Min', psir_images_df['R-L_resolution'].min(),
                                psir_images_df['A-P_resolution'].min(),
                                psir_images_df['I-S_resolution'].min()])
    table_resolution_psir.add_row(['Median', psir_images_df['R-L_resolution'].median(),
                                psir_images_df['A-P_resolution'].median(),
                                psir_images_df['I-S_resolution'].median()])
    table_resolution_psir.add_row(['Std', round(psir_images_df['R-L_resolution'].std(), 2),
                                round(psir_images_df['A-P_resolution'].std(), 2),
                                round(psir_images_df['I-S_resolution'].std(), 2)])
    print("PSIR Resolution Statistics:")
    print(table_resolution_psir)

    # Filter only STIR images
    stir_images_df = images_df[images_df['contrast'] == 'STIR']
    # Build a table showing the distribution of resolution
    table_resolution_stir = PrettyTable()
    table_resolution_stir.field_names = ['', 'R-L Resolution', 'A-P Resolution', 'I-S Resolution']
    ## In this case we want max, mean, median, min and std of the resolution
    table_resolution_stir.add_row(['Mean', round(stir_images_df['R-L_resolution'].mean(), 2),
                                round(stir_images_df['A-P_resolution'].mean(), 2),
                                round(stir_images_df['I-S_resolution'].mean(), 2)])
    table_resolution_stir.add_row(['Max', stir_images_df['R-L_resolution'].max(),
                                stir_images_df['A-P_resolution'].max(),
                                stir_images_df['I-S_resolution'].max()])
    table_resolution_stir.add_row(['Min', stir_images_df['R-L_resolution'].min(),
                                stir_images_df['A-P_resolution'].min(),
                                stir_images_df['I-S_resolution'].min()])
    table_resolution_stir.add_row(['Median', stir_images_df['R-L_resolution'].median(),
                                stir_images_df['A-P_resolution'].median(),
                                stir_images_df['I-S_resolution'].median()])
    table_resolution_stir.add_row(['Std', round(stir_images_df['R-L_resolution'].std(), 2),
                                round(stir_images_df['A-P_resolution'].std(), 2),
                                round(stir_images_df['I-S_resolution'].std(), 2)])
    print("STIR Resolution Statistics:")
    print(table_resolution_stir)

    # Filter only MP2RAGE images
    mp2rage_images_df = images_df[images_df['contrast'] == 'MP2RAGE']
    # Build a table showing the distribution of resolution
    table_resolution_mp2rage = PrettyTable()
    table_resolution_mp2rage.field_names = ['', 'R-L Resolution', 'A-P Resolution', 'I-S Resolution']
    ## In this case we want max, mean, median, min and std of the resolution
    table_resolution_mp2rage.add_row(['Mean', round(mp2rage_images_df['R-L_resolution'].mean(), 2),
                                round(mp2rage_images_df['A-P_resolution'].mean(), 2),
                                round(mp2rage_images_df['I-S_resolution'].mean(), 2)])
    table_resolution_mp2rage.add_row(['Max', mp2rage_images_df['R-L_resolution'].max(),
                                mp2rage_images_df['A-P_resolution'].max(),
                                mp2rage_images_df['I-S_resolution'].max()])
    table_resolution_mp2rage.add_row(['Min', mp2rage_images_df['R-L_resolution'].min(),
                                mp2rage_images_df['A-P_resolution'].min(),
                                mp2rage_images_df['I-S_resolution'].min()])
    table_resolution_mp2rage.add_row(['Median', mp2rage_images_df['R-L_resolution'].median(),
                                mp2rage_images_df['A-P_resolution'].median(),
                                mp2rage_images_df['I-S_resolution'].median()])
    table_resolution_mp2rage.add_row(['Std', round(mp2rage_images_df['R-L_resolution'].std(), 2),
                                round(mp2rage_images_df['A-P_resolution'].std(), 2),
                                round(mp2rage_images_df['I-S_resolution'].std(), 2)])
    print("MP2RAGE Resolution Statistics:")
    print(table_resolution_mp2rage)

    # Save the tables to the output folder in a txt file
    txt_file = os.path.join(output_folder, 'dataset_analysis.txt')
    os.makedirs(output_folder, exist_ok=True)
    with open(txt_file, 'w') as f:
        f.write("Dataset Analysis\n")
        f.write("================\n\n")
        f.write("Manufacturer Distribution:\n")
        f.write(str(table_manufacturer) + "\n\n")
        f.write("MRI Power Distribution:\n")
        f.write(str(table_mri_power) + "\n\n")
        f.write("Contrast Distribution:\n")
        f.write(str(table_contrast) + "\n\n")
        f.write("Resolution Statistics:\n")
        f.write(str(table_resolution) + "\n")
        f.write("\nT2w Resolution Statistics:\n")
        f.write(str(table_resolution_t2) + "\n")
        f.write("\nPSIR Resolution Statistics:\n")
        f.write(str(table_resolution_psir) + "\n")
        f.write("\nSTIR Resolution Statistics:\n")
        f.write(str(table_resolution_stir) + "\n")
        f.write("\nMP2RAGE Resolution Statistics:\n")
        f.write(str(table_resolution_mp2rage) + "\n")


if __name__ == '__main__':
    main()