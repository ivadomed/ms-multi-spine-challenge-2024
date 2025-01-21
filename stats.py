import os
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory of your BIDS dataset
bids_root = "ms-multi-spine-challenge-2024"

# Function to reorient to LAS and extract resolution, shape, and orientation
def extract_and_reorient_scan_info(file_path):
    img = nib.load(file_path)
    reoriented_img = nib.as_closest_canonical(img)  # Reorient to LAS
    shape = reoriented_img.shape
    resolution = reoriented_img.header.get_zooms()
    orientation = nib.aff2axcodes(reoriented_img.affine)
    return shape, resolution, orientation

# Collect scan info
scan_info = []
modalities = ["T2", "STIR", "PSIR", "MP2RAGE"]

for root, dirs, files in os.walk(bids_root):
    for dir in dirs:
        if 'sub' in dir:
            for file in os.listdir(os.path.join(root, dir, 'anat')):
                if file.endswith(".nii.gz") and "preproc" not in file:
                    for modality in modalities:
                        if modality in file:
                            file_path = os.path.join(root, dir, 'anat', file)
                            if 'derivatives' not in file_path:
                                shape, resolution, orientation = extract_and_reorient_scan_info(file_path)
                                scan_info.append({
                                    "File": file_path,
                                    "Modality": modality,
                                    "Shape": shape,
                                    "Resolution (mm)": resolution,
                                    "Orientation": orientation
                                })

# Convert to DataFrame
df = pd.DataFrame(scan_info)

# Save raw data to CSV
output_csv = "scan_statistics_reoriented.csv"
df.to_csv(output_csv, index=False)

# Generate histograms for each modality and axis
os.makedirs("histograms", exist_ok=True)

# Axis labels in LAS convention
axis_labels = ["L-R", "A-P", "S-I"]

for modality in modalities:
    modality_df = df[df["Modality"] == modality]

    for axis, axis_label in enumerate(axis_labels):
        # Resolution Data
        resolution_values = modality_df["Resolution (mm)"].apply(lambda x: x[axis]).values
        resolution_median = np.median(resolution_values)
        resolution_mean = np.mean(resolution_values)

        # Resolution Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(
            resolution_values,
            bins=30,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
        plt.title(
            f"Resolution ({axis_label}) - {modality}\n"
            f"Mean: {resolution_mean:.2f} mm, Median: {resolution_median:.2f} mm",
            fontsize=14
        )
        plt.xlabel(f"Resolution (mm) - {axis_label}", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"histograms/{modality}_resolution_{axis_label}.png")
        plt.close()

        # Shape Data
        shape_values = modality_df["Shape"].apply(lambda x: x[axis]).values
        shape_median = np.median(shape_values)
        shape_mean = np.mean(shape_values)

        # Shape Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(
            shape_values,
            bins=30,
            color='green',
            alpha=0.7,
            edgecolor='black'
        )
        plt.title(
            f"Shape ({axis_label}) - {modality}\n"
            f"Mean: {shape_mean:.2f}, Median: {shape_median:.2f}",
            fontsize=14
        )
        plt.xlabel(f"Shape (Voxels) - {axis_label}", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"histograms/{modality}_shape_{axis_label}.png")
        plt.close()

print(f"Reoriented statistics saved to {output_csv}")
print("Histograms saved in 'histograms/' folder.")
