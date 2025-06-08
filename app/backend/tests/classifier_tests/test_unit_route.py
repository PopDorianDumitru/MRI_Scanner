import io
import gzip
import numpy as np
import nibabel as nib
from unittest import mock
from fastapi.testclient import TestClient
from main import app
import tempfile
import os

client = TestClient(app)


def create_dummy_nii_gz():
    # Create dummy MRI data
    data = np.random.rand(64, 64, 64)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Save to a temporary .nii file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    # Read the .nii file and gzip it in memory
    with open(tmp_path, "rb") as f:
        uncompressed_data = f.read()
    os.remove(tmp_path)  # Clean up the temp file

    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz:
        gz.write(uncompressed_data)

    compressed_buffer.seek(0)
    return compressed_buffer

@mock.patch("api.classifier.classify_mri")
def test_classify_route_success(mock_classify_mri):
    dummy_file = create_dummy_nii_gz()

    # Mock classification output
    mock_classify_mri.return_value = {
        "label": "Mild Dementia",
        "image": "base64string=="
    }

    response = client.post(
        "/classify/",
        files={"file": ("test.nii.gz", dummy_file, "application/gzip")},
        data={"orientation": "Axial"}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert "label" in json_data
    assert "image" in json_data
    assert json_data["label"] == "Mild Dementia"


def test_classify_route_invalid_file_extension():
    response = client.post(
        "/classify/",
        files={"file": ("badfile.jpg", io.BytesIO(b"not nii.gz"), "image/jpeg")},
        data={"orientation": "Axial"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file format. Must be .nii.gz"


@mock.patch("api.classifier.classify_mri", side_effect=Exception("crash"))
def test_classify_route_internal_error(mock_classify_mri):
    dummy_file = create_dummy_nii_gz()

    response = client.post(
        "/classify/",
        files={"file": ("test.nii.gz", dummy_file, "application/gzip")},
        data={"orientation": "Axial"}
    )

    assert response.status_code == 500
    assert "Processing failed" in response.json()["detail"]
