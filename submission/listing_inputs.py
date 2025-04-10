import os
import re
import argparse


# Define expected file path patterns using regular expressions.
EXPECTED_PATHS_PATTERNS = [
    r'rawdata/sub-\d{3}/\d{2}-\d{3}_T2.nii.gz',
    r'rawdata/sub-\d{3}/\d{2}-\d{3}_STIR.nii.gz',
    r'rawdata/sub-\d{3}/\d{2}-\d{3}_PSIR.nii.gz',
    r'rawdata/sub-\d{3}/\d{2}-\d{3}_MP2RAGE.nii.gz',
    r'derivatives/preprocessed/sub-\d{3}/\d{2}-\d{3}_T2.nii.gz',
    r'derivatives/preprocessed/sub-\d{3}/\d{2}-\d{3}_STIR.nii.gz',
    r'derivatives/preprocessed/sub-\d{3}/\d{2}-\d{3}_PSIR.nii.gz',
    r'derivatives/preprocessed/sub-\d{3}/\d{2}-\d{3}_MP2RAGE.nii.gz',
    r'derivatives/preprocessedAndRegistered/sub-\d{3}/\d{2}-\d{3}_T2.nii.gz',
    r'derivatives/preprocessedAndRegistered/sub-\d{3}/\d{2}-\d{3}_STIR.nii.gz',
    r'derivatives/preprocessedAndRegistered/sub-\d{3}/\d{2}-\d{3}_PSIR.nii.gz',
    r'derivatives/preprocessedAndRegistered/sub-\d{3}/\d{2}-\d{3}_MP2RAGE.nii.gz',
]


def find_pattern(pattern: str, path: str, followlinks: bool = False) -> list[str]:
    """ Recursively search for files in the given directory path that match the provided pattern."""
    result = []
    for root, dirs, files in os.walk(path, followlinks=followlinks):
        for name in files:
            if re.search(pattern, name):
                result.append(os.path.join(root, name))
    return result


def listing_input_files(path: str) -> list[str]:
    """ Verify the presence of expected files in the directory and return the list of their paths. """
    input_files = []
    existing_files = find_pattern(
        pattern=r'\d{2}-\d{3}_(T2|STIR|PSIR|MP2RAGE).nii.gz',
        path=path,
    )

    for expected_pattern in EXPECTED_PATHS_PATTERNS:
        file_found = False
        for file_path in existing_files:
            if bool(re.search(expected_pattern, file_path)):
                input_files.append(file_path)
                file_found = True
        if not file_found:
            input_files.append(None)

    return input_files


def main(args):
    parsed_list = listing_input_files(args.input_folder)
    print(parsed_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script dedicated to parsing an inference step data folder."
    )
    parser.add_argument('-i', '--input_folder', required=False, default= './test_data/test_case_1', help="Path to the input folder.")
    args = parser.parse_args()

    main(args)
