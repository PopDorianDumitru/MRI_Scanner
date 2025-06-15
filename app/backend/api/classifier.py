from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pickle
import shutil
import tempfile
import os
from PIL import Image
import nibabel as nib
import numpy as np
import torch
from R3GAN.MRI2DCNN import MRI2DCNN
from torchvision import transforms
from config import CLASSIFIER_MODEL_PATH
from services.classifier_service import classify_mri, classify_image

router = APIRouter()
classifier_path = CLASSIFIER_MODEL_PATH
VALID_ORIENTATIONS = {"sagittal", "coronal", "axial"}

with open(classifier_path, "rb") as f:
    state_dict = pickle.load(f)
classifier = MRI2DCNN(num_classes=5)
classifier.load_state_dict(state_dict)
classifier.eval()

inference_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@router.post("/")  
async def classify_route(
    file: UploadFile = File(...),
    orientation: str = Form(...)
):
    if not file.filename.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Invalid file format. Must be .nii.gz")

    try:
        prediction = classify_mri(file, orientation)
        return JSONResponse(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
@router.post("/image")
async def classify_image_route(
    file: UploadFile = File(...)
):
    # Basic image format check
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format. Must be .png or .jpg")

    try:
        prediction = classify_image(file)
        return JSONResponse(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")