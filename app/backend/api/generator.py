from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import io
import numpy as np
import base64
from R3GAN.model import open_model
from R3GAN.gen_images import generate_images_ui
from config import GENERATOR_MODEL_PATH
from services.generator_service import generate_images_for_label
from services.generator_service import InvalidCountError, InvalidLabelError

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
    try:
        images = generate_images_for_label(label, count)
    except InvalidLabelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidCountError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse({"images": images})