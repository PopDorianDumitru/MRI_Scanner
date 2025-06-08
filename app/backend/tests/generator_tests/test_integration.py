import pytest
from unittest import mock
from fastapi.testclient import TestClient
from main import app  # adjust if your FastAPI app is created elsewhere
from PIL import Image
import io

client = TestClient(app)


# Helper to create mock PIL images
def create_dummy_pil_images(count):
    return [Image.new("RGB", (128, 128), color=(i*20, i*20, i*20)) for i in range(count)]


@mock.patch("services.generator_service.generate_images_ui")
def test_generate_route_success(mock_generate_images_ui):
    mock_generate_images_ui.return_value = create_dummy_pil_images(3)

    response = client.post(
        "/generate/",
        data={"label": "Mild Dementia", "count": 3}
    )

    assert response.status_code == 200
    body = response.json()

    assert "images" in body
    assert isinstance(body["images"], list)
    assert len(body["images"]) == 3
    for img in body["images"]:
        assert isinstance(img, str)
        assert img.startswith("iVBOR")  # base64 of PNG header


@mock.patch("services.generator_service.generate_images_ui")
def test_generate_route_invalid_label(mock_generate_images_ui):
    response = client.post(
        "/generate/",
        data={"label": "Fake Dementia", "count": 2}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid dementia label."


@mock.patch("services.generator_service.generate_images_ui")
def test_generate_route_invalid_count(mock_generate_images_ui):
    response = client.post(
        "/generate/",
        data={"label": "Normal", "count": 999}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Count must be between 0 and 5."
