import argparse

from Preprocessor import Preprocessor


def main(path_to_diagnosis_csv, path_to_mri_scans_folder, path_to_output, slice):
    # Example starting print to show arguments received
    print(f"Diagnosis CSV path: {path_to_diagnosis_csv}")
    print(f"MRI scans folder path: {path_to_mri_scans_folder}")
    print(f"Output folder path: {path_to_output}")
    print(f"Slice: {slice}")

    print("Preparing dataset...")
    Preprocessor.prepare_dataset(path_to_diagnosis_csv, path_to_mri_scans_folder, path_to_output, slice)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MRI dataset from diagnosis and scan folders.")
    parser.add_argument("csv", type=str, help="Path to the diagnosis CSV file (e.g., CDR data).")
    parser.add_argument("scans", type=str, help="Path to the folder containing MRI scan folders.")
    parser.add_argument("output", type=str, help="Path to the folder output.")
    parser.add_argument("slice", type=str, help="Type of slice.")

    args = parser.parse_args()

    main(args.csv, args.scans, args.output, args.slice)
