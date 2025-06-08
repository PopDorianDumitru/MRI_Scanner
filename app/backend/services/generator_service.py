import base64
import io
import numpy as np
from typing import Literal
from config import GENERATOR_MODEL_PATH
from R3GAN.gen_images import generate_images_ui

VALID_LABELS = {"Normal", "Mild Dementia", "Moderate Dementia", "Severe Dementia", "Very Severe Dementia"}
label_to_index = {
    "Normal": 0,
    "Mild Dementia": 1,
    "Moderate Dementia": 2,
    "Severe Dementia": 3,
    "Very Severe Dementia": 4
}

class InvalidLabelError(ValueError):
    pass

class InvalidCountError(ValueError):
    pass


def generate_images_for_label(label: str, count: int) -> list[str]:
    if label not in VALID_LABELS:
        raise InvalidLabelError("Invalid dementia label.")
    if not (0 <= count <= 5):
        raise InvalidCountError("Count must be between 0 and 5.")

    seeds = np.random.randint(0, 1e6, size=count).tolist()
    class_idx = label_to_index[label]
    images = generate_images_ui(GENERATOR_MODEL_PATH, seeds, class_idx)

    encoded_images = []
    for img_pil in images:
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encoded_images.append(img_base64)

    return encoded_images
