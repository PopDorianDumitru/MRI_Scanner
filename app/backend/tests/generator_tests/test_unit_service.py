import pytest
from unittest import mock
from PIL import Image
from services.generator_service import (
    generate_images_for_label,
    InvalidLabelError,
    InvalidCountError
)


@pytest.fixture
def dummy_images():
    # Create mock PIL images
    return [Image.new("RGB", (128, 128), color=(i*50, i*50, i*50)) for i in range(3)]


@mock.patch("services.generator_service.generate_images_ui")
def test_generate_images_valid(mock_generate_images_ui, dummy_images):
    mock_generate_images_ui.return_value = dummy_images

    label = "Mild Dementia"
    count = 3
    results = generate_images_for_label(label, count)

    assert isinstance(results, list)
    assert len(results) == 3
    for img in results:
        assert isinstance(img, str)
        assert img.startswith("iVBOR")  # PNG Base64 header


def test_generate_images_invalid_label():
    with pytest.raises(InvalidLabelError, match="Invalid dementia label."):
        generate_images_for_label("Invalid Label", 2)


def test_generate_images_invalid_count_too_low():
    with pytest.raises(InvalidCountError, match="Count must be between 0 and 5."):
        generate_images_for_label("Normal", -1)


def test_generate_images_invalid_count_too_high():
    with pytest.raises(InvalidCountError, match="Count must be between 0 and 5."):
        generate_images_for_label("Normal", 10)
