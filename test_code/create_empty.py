import nibabel as nib
import numpy as np

# Path to one of your actual T2 scans
example_t2_path = "dataset_split/train/sub-001/anat/sub-001_T2w.nii.gz"

# Load the T2 scan
example_t2 = nib.load(example_t2_path)

# Create a zero-filled version of the data
zero_filled_data = np.zeros(example_t2.shape, dtype=example_t2.get_data_dtype())

# Create a new NIfTI image with the same affine and header as the original T2 scan
default_nifti = nib.Nifti1Image(zero_filled_data, affine=example_t2.affine, header=example_t2.header)

# Save the NIfTI image
output_path = "default.nii.gz"
nib.save(default_nifti, output_path)

output_path
