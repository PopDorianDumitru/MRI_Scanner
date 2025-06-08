from fastapi.testclient import TestClient
from unittest import mock
from main import app
from tests.utils import generate_valid_nii_gz_file, generate_invalid_2d_nii_gz_file

client = TestClient(app)

@mock.patch("services.classifier_service.classifier")
def test_integration_route_service(mock_classifier):
    # Mock the model's output to return class index 2 (Mild Dementia)
    mock_output = mock.Mock()
    mock_output.argmax.return_value.item.return_value = 2
    mock_classifier.return_value = mock_output

    file_data = generate_valid_nii_gz_file()

    response = client.post(
        "/classify",
        files={"file": ("test.nii.gz", file_data, "application/gzip")},
        data={"orientation": "Axial"}
    )

    assert response.status_code == 200
    json_data = response.json()

    assert json_data["label"] == "Mild Dementia"
    assert json_data["image"].startswith("iVBOR")  # base64 PNG

def test_classify_invalid_extension():
    response = client.post(
        "/classify",
        files={"file": ("invalid.jpg", b"fake-content", "image/jpeg")},
        data={"orientation": "Axial"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file format. Must be .nii.gz"


@mock.patch("services.classifier_service.classifier")
def test_classify_invalid_orientation(mock_classifier):
    file_data = generate_valid_nii_gz_file()
    response = client.post(
        "/classify",
        files={"file": ("test.nii.gz", file_data, "application/gzip")},
        data={"orientation": "Diagonal"}
    )
    assert response.status_code == 500
    assert "Invalid orientation" in response.json()["detail"]


@mock.patch("services.classifier_service.classifier")
def test_classify_invalid_volume_shape(mock_classifier):
    file_data = generate_invalid_2d_nii_gz_file()
    response = client.post(
        "/classify",
        files={"file": ("test.nii.gz", file_data, "application/gzip")},
        data={"orientation": "Axial"}
    )
    assert response.status_code == 500
    assert "Input volume must be a 3D array" in response.json()["detail"]