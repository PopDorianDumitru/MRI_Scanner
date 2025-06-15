import pytest
from unittest import mock
from fastapi.testclient import TestClient
from main import app
from services.generator_service import InvalidLabelError, InvalidCountError
from PIL import Image

client = TestClient(app)


@mock.patch("api.generator.generate_images_for_label")
def test_generate_route_success(mock_generate):
    # Mock return value
    mock_generate.return_value = ["base64img1==", "base64img2=="]

    response = client.post(
        "/generate/",
        data={"label": "Moderate Dementia", "count": 2}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert "images" in json_data
    assert len(json_data["images"]) == 2
    assert json_data["images"][0].startswith("base64")


@mock.patch("api.generator.generate_images_for_label", side_effect=InvalidLabelError("Invalid dementia label."))
def test_generate_route_invalid_label(mock_generate):
    response = client.post(
        "/generate/",
        data={"label": "Alien Dementia", "count": 1}
    )

    assert response.status_code == 400
    assert "Invalid dementia label." in response.json()["detail"]


@mock.patch("api.generator.generate_images_for_label", side_effect=InvalidCountError("Count must be between 0 and 5."))
def test_generate_route_invalid_count(mock_generate):
    response = client.post(
        "/generate/",
        data={"label": "Normal", "count": 10}
    )

    assert response.status_code == 400
    assert "Count must be between 0 and 5." in response.json()["detail"]
