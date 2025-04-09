import os
import numpy as np
import nibabel as nib
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
from nibabel.orientations import (
    io_orientation, axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes
)


class Preprocessor:
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
        img.save(os.path.join(output_directory, img_name))

    @staticmethod
    def convert_slice_to_image_file(mri_slice, size=(128, 128)):
        norm = Preprocessor.normalize_slice(mri_slice)
        img = Image.fromarray(norm)
        img = img.resize(size)
        return img

    @staticmethod
    def extract_center_slices(volume, num_slices=30):
        if volume.ndim != 3:
            raise ValueError("Input volume must be a 3D array.")
        z = volume.shape[2]
        center = z // 2
        half = num_slices // 2
        start = max(center - half, 0)
        end = min(center + half, z)
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