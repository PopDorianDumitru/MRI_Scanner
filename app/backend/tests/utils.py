import numpy as np
import nibabel as nib
import gzip
import tempfile
import io
import os

def generate_valid_nii_gz_file() -> io.BytesIO:
    data = np.random.rand(32, 32, 32)  # Small, fast to process
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Save to a temporary NIfTI file
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    # Compress to .nii.gz
    with open(tmp_path, "rb") as f:
        raw = f.read()
    os.remove(tmp_path)

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb') as gz:
        gz.write(raw)
    compressed.seek(0)
    return compressed


def generate_invalid_2d_nii_gz_file():
    data = np.random.rand(64, 64)  # Only 2D
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        raw = f.read()
    os.remove(tmp_path)

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb') as gz:
        gz.write(raw)
    compressed.seek(0)
    return compressed
