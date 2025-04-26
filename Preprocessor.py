import json
import os
import random

import numpy as np
import nibabel as nib
from PIL import Image
from collections import defaultdict
import shutil
from scipy.ndimage import affine_transform
import pandas as pd
import xml.etree.ElementTree as ET
from nibabel.orientations import (
    io_orientation, axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes, inv_ornt_aff
)


class Preprocessor:
    path_to_diagnosis_csv = "C:\\Users\\doria\\Desktop\\Licenta\\Dataset_TAR\\OASIS3_data_files\\UDSb4\\csv\\OASIS3_UDSb4_cdr.csv"
    path_to_mri_scans_folder = "C:\\Users\\doria\\Desktop\\Licenta\\drive-download-20250425T223131Z-001"
    path_to_output_folder = "C:\\Users\\doria\\Desktop\\Licenta\\output"
    patient_ids = []
    session_ids = []
    cdr_rating_sessions = defaultdict(list)
    orientation_map = {
        'SAG': ('L', 'S', 'A'),
        'COR': ('L', 'A', 'S'),
        'AXIAL': ('L', 'P', 'S')  # or 'TRA'
    }

    labels_map = {
        'SAG': (('L', 'R'), ('I', 'S'), ('P', 'A')),
        'COR': (('L', 'R'), ('P', 'A'), ('I', 'S')),
        'AXIAL': (('L', 'R'), ('P', 'A'), ('I', 'S'))  # same as default RAS
    }

    @staticmethod
    def normalize_slice(mri_slice):
        mri_slice = mri_slice - np.min(mri_slice)
        mri_slice = mri_slice / (np.max(mri_slice) + 1e-5)
        mri_slice = (mri_slice * 255).astype(np.uint8)
        return mri_slice

    @staticmethod
    def save_image(img, img_name, output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f'Saving image ${img_name} in folder: ${output_directory}')
        img.save(os.path.join(output_directory, img_name))

    @staticmethod
    def convert_slice_to_image_file(mri_slice, size=(128, 128)):
        norm = Preprocessor.normalize_slice(mri_slice)
        img = Image.fromarray(norm)
        img = img.resize(size)
        return img

    @classmethod
    def extract_center_slices(cls, volume, num_slices=30):
        print("Trying to extract center slices")

        # Make sure you are slicing the array, not the Nifti1Image
        if hasattr(volume, 'get_fdata'):
            volume = volume.get_fdata()

        if volume.ndim != 3:
            raise ValueError("Input volume must be a 3D array.")

        start, end = cls.find_brain_region_slices(volume)
        half = (start + end) // 2
        z = volume.shape[2]
        start = half
        end = min(start + num_slices, z)

        return [volume[:, :, i] for i in range(start, end)]

    @staticmethod
    def find_hdr_files_per_subject_raw(subject_folder):
        hdr_files = []
        for subject_data in os.listdir(subject_folder):
            if subject_data == "RAW":
                raw_path = os.path.join(subject_folder, "RAW")
                if os.path.isdir(raw_path):
                    for file in os.listdir(raw_path):
                        if file.lower().endswith(".hdr"):
                            hdr_files.append(os.path.join(raw_path, file))
        return sorted(hdr_files)

    @staticmethod
    def find_gz_files_per_subject_raw(subject_folder):
        gz_files = []
        print(subject_folder)
        # Walk through all subfolders and files
        for root, dirs, files in os.walk(subject_folder):
            for file in files:
                print(file)
                if file.endswith('.nii.gz'):
                    full_path = os.path.join(root, file)
                    gz_files.append(full_path)

        return gz_files[0]

    @staticmethod
    def get_scan_orientation(subject_folder):
        for file in os.listdir(subject_folder):
            if file.endswith('.xml'):
                xml_path = os.path.join(subject_folder, file)
                break
        else:
            raise FileNotFoundError("No XML file found in subject folder.")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag.lower().endswith('orientation'):
                return elem.text.strip().upper()
        return None

    @classmethod
    def load_and_reorient_to_axial(cls, hdr_path, original_orientation):
        img = nib.load(hdr_path)
        if original_orientation not in cls.orientation_map:
            raise ValueError("Not valid orientation")
        data = img.get_fdata().squeeze()
        starting_codes = cls.labels_map.get(original_orientation)
        starting_orientation = axcodes2ornt(aff2axcodes(img.affine), starting_codes)
        target_orientation = axcodes2ornt(('R', 'A', 'S'))
        transform = ornt_transform(starting_orientation, target_orientation)
        reoriented_data = apply_orientation(data, transform)
        return reoriented_data

    @classmethod
    def process_subject(cls, subject_folder, num_slices=30, output_size=(128, 128), output_path="."):
        hdr_paths = cls.find_hdr_files_per_subject_raw(subject_folder)
        if not hdr_paths:
            raise ValueError(f"No HDR files found in {subject_folder}.")

        orientation = cls.get_scan_orientation(subject_folder)
        volume = cls.load_and_reorient_to_axial(hdr_paths[0], orientation)
        slices = cls.extract_center_slices(volume, num_slices=num_slices)

        subject_name = os.path.basename(subject_folder.strip("/"))
        subject_output_dir = os.path.join(output_path, subject_name)
        os.makedirs(subject_output_dir, exist_ok=True)

        for idx, mri_slice in enumerate(slices):
            image = cls.convert_slice_to_image_file(mri_slice, size=output_size)
            image_name = f"{subject_name}_slice_{idx:02d}.png"
            cls.save_image(image, image_name, subject_output_dir)
        return len(slices)


    @classmethod
    def process_all_subjects_in_directory(cls, parent_folder, output_path="processed", num_slices=15, output_size=(128, 128)):
        os.makedirs(output_path, exist_ok=True)
        for folder_name in os.listdir(parent_folder):
            subject_folder = os.path.join(parent_folder, folder_name)
            if os.path.isdir(subject_folder):
                try:
                    print(f"Processing: {subject_folder}")
                    cls.process_subject(subject_folder, num_slices=num_slices, output_size=output_size, output_path=output_path)
                except Exception as e:
                    print(f"Failed to process {subject_folder}: {e}")

    @staticmethod
    def flatten_image_directory(source_root, output_folder, allowed_ext=(".png", ".jpg", ".jpeg")):
        os.makedirs(output_folder, exist_ok=True)
        counter = 0

        for dirpath, _, filenames in os.walk(source_root):
            for filename in filenames:
                if filename.lower().endswith(allowed_ext):
                    src = os.path.join(dirpath, filename)
                    # Unique filename to avoid overwriting
                    dst_filename = f"img_{counter:05d}{os.path.splitext(filename)[1]}"
                    dst = os.path.join(output_folder, dst_filename)
                    shutil.copy2(src, dst)
                    counter += 1

        print(f"Copied {counter} images to {output_folder}")

    @classmethod
    def load_patient_data(self):
        print(f'Trying to load data from ${self.path_to_diagnosis_csv}')
        try:
            df = pd.read_csv(self.path_to_diagnosis_csv)
            patient_id_column = "OASISID"
            session_id_column = "OASIS_session_label"
            cdr_score_column = "CDRTOT"
            for index, row in df.iterrows():
                patient_id = row[patient_id_column]
                session_id = row[session_id_column]
                cdr_score = row[cdr_score_column]
                self.patient_ids.append(patient_id)
                self.session_ids.append(session_id)
                self.cdr_rating_sessions[cdr_score].append(session_id)
            print(f'Successfully loaded data from ${self.path_to_diagnosis_csv}')
        except Exception as e:
            print(f'Failed to load data from ${self.path_to_diagnosis_csv}: {e}')

    @classmethod
    def choose_random_sessions(cls, cdr_scores, chosen_samples):
        random_sessions = [None, None, None, None, None]
        for index in range(len(cdr_scores)):
            random_sessions[index] = random.sample(cls.cdr_rating_sessions[cdr_scores[index]], chosen_samples[index])
        return random_sessions

    @classmethod
    def get_patient_ids_to_mri_scan_days(cls):
        patient_sessions = defaultdict(list)

        # List all folders inside the dataset path
        for folder_name in os.listdir(cls.path_to_mri_scans_folder):
            folder_path = os.path.join(cls.path_to_mri_scans_folder, folder_name)

            if os.path.isdir(folder_path):
                # Example folder name: OAS30001_MR_d0129
                split_name = folder_name.split("_")
                patient_id = split_name[0]
                day_number = int(split_name[2][1:])
                patient_sessions[patient_id].append(day_number)
        return patient_sessions

    @classmethod
    def prepare_folders(cls, cdr_scores):
        # Make sure output path exists
        os.makedirs(cls.path_to_output_folder, exist_ok=True)
        folders = []
        for score in cdr_scores:
            folder_name = f"cdr_{str(score).replace('.', '_')}"  # Replace '.' with '_' for folder names
            folders.append(folder_name)
            folder_path = os.path.join(cls.path_to_output_folder, folder_name)

            os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        print(f"Prepared folders for CDR scores: {cdr_scores}")
        return folders

    @classmethod
    def prepare_dataset(cls, csv_path, scans_path, output_path):
        cdr_scores = [0, 0.5, 1, 2, 3]
        chosen_samples = [125, 125, 125, 125, 19]
        nr_of_slices = [20, 20, 20, 20, 55]
        scans_sessions = [[], [], [], [], []]

        cls.path_to_diagnosis_csv = csv_path
        cls.path_to_mri_scans_folder = scans_path
        cls.path_to_output_folder = output_path

        cls.load_patient_data()
        folders = cls.prepare_folders(cdr_scores)

        patients_scans = cls.get_patient_ids_to_mri_scan_days()
        sessions_by_severity = cls.choose_random_sessions(cdr_scores, chosen_samples)

        print(sessions_by_severity[4])
        # Now in scans_sessions you have a list containing the list of scans for each label
        cls.match_sessions_to_scans(sessions_by_severity, patients_scans, scans_sessions)
        index = 0
        for scan_sessions in scans_sessions:
            for scan_session in scan_sessions:
                cls.process_subject_gz(scan_session, nr_of_slices[index], output_path=os.path.join(cls.path_to_output_folder, folders[index]))
            index += 1

    @classmethod
    def process_subject_gz(cls, subject_folder, num_slices=30, output_size=(128, 128), output_path="."):
        gz_path = cls.find_gz_files_per_subject_raw(os.path.join(cls.path_to_mri_scans_folder, subject_folder))
        if not gz_path:
            raise ValueError(f"No gz files found in {subject_folder}.")

            # Step 1: extract filename without extension
        filename = os.path.basename(gz_path).replace('.nii.gz', '')

        # Step 2: replace NIFTI with BIDS in path
        bids_folder = gz_path.replace('/NIFTI/', '/BIDS/').split('/' + filename)[0]

        # Step 3: create the path to the .json file
        json_path = os.path.join(bids_folder, filename + '.json')

        # Step 4: load JSON metadata
        if not os.path.exists(json_path):
            raise ValueError(f"JSON metadata file not found: {json_path}")

        with open(json_path, 'r') as f:
            json_metadata = json.load(f)

        img = nib.load(gz_path)
        data = img.get_fdata()
        affine = img.affine  # original affine

        if cls.needs_rotation(json_metadata):
            print(f'{subject_folder} reoriented')
            data = np.transpose(data, (2, 0, 1))
            data = np.flip(data, axis=1)
            new_img = nib.Nifti1Image(data, np.eye(4))
        else:
            new_img = nib.Nifti1Image(data, affine)
            # VERY IMPORTANT: canonicalize AFTER rotation
            new_img = nib.as_closest_canonical(new_img)

        slices = cls.extract_center_slices(new_img, num_slices=num_slices)

        for idx, mri_slice in enumerate(slices):
            image = cls.convert_slice_to_image_file(mri_slice, size=output_size)
            image_name = f"{subject_folder}_slice_{idx:02d}.png"
            cls.save_image(image, image_name, output_path)
        return len(slices)

    @classmethod
    def needs_rotation(cls, json_metadata: dict):
        orientation = json_metadata.get("ImageOrientationPatientDICOM")
        if orientation == None:
            return True

        return False
    @classmethod
    def match_sessions_to_scans(cls, session_ids_by_severity, patients_scans, scans_sessions):
        index = 0
        for session_ids in session_ids_by_severity:
            for session_id in session_ids:
                # Example folder name: OAS30001_MR_d0129
                split_id = session_id.split("_")
                patient_id = split_id[0]
                number_of_days = int(split_id[2][1:])
                scan_days = patients_scans.get(patient_id, [])
                closest_scan_day = min(scan_days, key=lambda x: abs(x - number_of_days))

                # Check if within 365 days tolerance
                day_number = ""
                if closest_scan_day < 10:
                    day_number = "000"
                elif closest_scan_day < 100:
                    day_number = "00"
                elif closest_scan_day < 1000:
                    day_number = "0"
                scans_sessions[index].append(patient_id + "_MR_d" + day_number + str(closest_scan_day))
            index += 1

    @classmethod
    def find_brain_region_slices(cls, data, threshold_ratio=0.1, margin=5):
        """
        Detect brain-containing slices along Z-axis based on intensity.

        Args:
            data: 3D MRI data (X, Y, Z).
            threshold_ratio: fraction of max intensity to consider as brain.
            margin: number of slices to expand at each end (to avoid cutting off useful brain).

        Returns:
            start_slice, end_slice: indices of brain region.
        """
        # Move along Z-axis
        slice_sums = np.array([
            np.sum(slice_ > (threshold_ratio * np.max(data)))
            for slice_ in data.transpose(2, 0, 1)
        ])

        brain_slices = np.where(slice_sums > 0)[0]

        if len(brain_slices) == 0:
            raise ValueError("No brain detected! Maybe threshold too high or corrupted scan.")

        start = max(brain_slices[0] - margin, 0)
        end = min(brain_slices[-1] + margin, data.shape[2])

        return start, end

# path_to_diagnosis_csv = "C:\\Users\\doria\\Desktop\\Licenta\\Dataset_TAR\\OASIS3_data_files\\UDSb4\\csv\\OASIS3_UDSb4_cdr.csv"
# path_to_mri_scans_folder = "C:\\Users\\doria\\Desktop\\Licenta\\drive-download-20250425T223131Z-001"
# Preprocessor.prepare_dataset(path_to_diagnosis_csv, path_to_mri_scans_folder)