from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import io
import numpy as np
import base64
from R3GAN.model import open_model
from R3GAN.gen_images import generate_images_ui
from config import GENERATOR_MODEL_PATH

router = APIRouter()
VALID_LABELS = {"Normal", "Mild Dementia", "Moderate Dementia", "Severe Dementia", "Very Severe Dementia"}
label_to_index = {
    "Normal": 0,
    "Mild Dementia": 1,
    "Moderate Dementia": 2,
    "Severe Dementia": 3,
    "Very Severe Dementia": 4
}
generator, _ = open_model(GENERATOR_MODEL_PATH)


@router.post("/")
async def generate_route(
    label: str = Form(...),
    count: int = Form(...)
):
    if label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid dementia label.")
    if not (0 <= count <= 5):
        raise HTTPException(status_code=400, detail="Count must be between 0 and 5.")

    seeds = np.random.randint(0, 1e6, size=count).tolist()
    class_idx = label_to_index[label]
    images = generate_images_ui(GENERATOR_MODEL_PATH, seeds, class_idx)
    encoded_images = []

    for img_pil in images:
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")  # save image as PNG into memory buffer
        buffer.seek(0)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        encoded_images.append(img_base64)

    return JSONResponse({"images": encoded_images})