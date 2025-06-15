import io
import pytest
import numpy as np
from unittest import mock
from types import SimpleNamespace
from services.classifier_service import classify_mri, classify_image
from PIL import Image

class DummyFile:
    def __init__(self, filename="test.nii.gz", data=None):
        self.filename = filename
        self.file = io.BytesIO(data or b"Dummy content")


@pytest.fixture
def dummy_mri_volume():
    return np.random.rand(64, 64, 64)


@mock.patch("services.classifier_service.nib.load")
@mock.patch("services.classifier_service.classifier")
def test_classify_mri_axial(mock_classifier, mock_nib_load, dummy_mri_volume):
    # Mock nib.load to return a dummy 3D volume
    mock_img = mock.Mock()
    mock_img.get_fdata.return_value = dummy_mri_volume
    mock_nib_load.return_value = mock_img

    # Mock classifier output
    mock_output = mock.Mock()
    mock_output.argmax.return_value.item.return_value = 2  # Mild Dementia

    # Set the classifier to return the mocked output when called
    mock_classifier.return_value = mock_output

    # Prepare dummy file
    dummy_file = DummyFile()

    result = classify_mri(dummy_file, orientation="Axial")

    assert result["label"] == "Mild Dementia"
    assert isinstance(result["image"], str)

@mock.patch("services.classifier_service.nib.load")
def test_classify_mri_invalid_orientation(mock_nib_load, dummy_mri_volume):
    dummy_file = DummyFile()
    mock_img = mock.Mock()
    mock_img.get_fdata.return_value = dummy_mri_volume
    mock_nib_load.return_value = mock_img

    error_message = "Invalid orientation. Must be Axial, Coronal, or Sagittal."

    with pytest.raises(ValueError, match=error_message):
        classify_mri(dummy_file, orientation="Invalid")


@mock.patch("services.classifier_service.nib.load")
def test_classify_mri_non_3d_input(mock_nib_load):
    dummy_file = DummyFile()

    mock_img = mock.Mock()
    mock_img.get_fdata.return_value = np.random.rand(64, 64)
    mock_nib_load.return_value = mock_img

    error_message = "Input volume must be a 3D array."

    with pytest.raises(ValueError, match=error_message):
        classify_mri(dummy_file, orientation="Axial")

@mock.patch("services.classifier_service.classifier")
def test_classify_image(mock_classifier):
    # Create a simple grayscale test image
    image = Image.new("L", (128, 128), color=128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Create a DummyFile-like object for the image
    file = SimpleNamespace(filename="test.png", file=buffer)

    # Mock classifier output
    mock_output = mock.Mock()
    mock_output.argmax.return_value.item.return_value = 1  # Slight Dementia
    mock_classifier.return_value = mock_output

    result = classify_image(file)

    assert result["label"] == "Slight Dementia"
    assert isinstance(result["image"], str)
    assert result["image"].startswith("iVBOR") or len(result["image"]) > 0 

def test_classify_image_invalid_file():
    bad_file = SimpleNamespace(
        filename="corrupt.png",
        file=io.BytesIO(b"This is not an image")
    )

    with pytest.raises(ValueError, match="Image classification failed:"):
        classify_image(bad_file)
