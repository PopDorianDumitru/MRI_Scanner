# services/classifier_service.py
from R3GAN.MRI2DCNN import MRI2DCNN
import os
import shutil
import tempfile
from typing import Literal
from torchvision import transforms
import base64
import nibabel as nib
import numpy as np
from PIL import Image
import torch
from fastapi import HTTPException
from config import CLASSIFIER_MODEL_PATH
import pickle 
import io

label_map = {
    0: "Normal",
    1: "Mild Dementia",
    2: "Moderate Dementia",
    3: "Severe Dementia",
    4: "Very Severe Dementia"
}

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


def classify_mri(file, orientation: Literal["Axial", "Coronal", "Sagittal"]) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        nii_path = os.path.join(tmpdir, file.filename)
        with open(nii_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        volume = nib.load(nii_path).get_fdata()

        if volume.ndim != 3:
            raise ValueError("Input volume must be a 3D array.")

        axis_map = {
            "Axial": 2,
            "Coronal": 1,
            "Sagittal": 0
        }
        axis = axis_map.get(orientation)
        if axis is None:
            raise ValueError("Invalid orientation. Must be Axial, Coronal, or Sagittal.")

        center = volume.shape[axis] // 2
        if orientation == "Axial":
            mri_slice = volume[:, :, center]
        elif orientation == "Coronal":
            mri_slice = volume[:, center, :]
        elif orientation == "Sagittal":
            mri_slice = volume[center, :, :]

        # Normalize and convert to image
        mri_slice = mri_slice - np.min(mri_slice)
        mri_slice = mri_slice / (np.max(mri_slice) + 1e-5)
        mri_slice = (mri_slice * 255).astype(np.uint8)
        img = Image.fromarray(mri_slice).resize((128, 128))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")


        input_tensor = inference_transform(img).unsqueeze(0)  # Shape: [1, 1, 128, 128]

        with torch.no_grad():
            outputs = classifier(input_tensor)
            predicted_idx = outputs.argmax(dim=1).item()

        return {
            "label": label_map.get(predicted_idx, "Unknown"),
            "image": base64_img
        }

def classify_image(file) -> dict:
    try:
        # Load and preprocess image using your defined transform
        image = Image.open(file.file).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0)  # [1, 1, 128, 128]

        # Create base64 image from the preprocessed (grayscale, resized) image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Run inference
        with torch.no_grad():
            outputs = classifier(input_tensor)
            predicted_idx = outputs.argmax(dim=1).item()

        return {
            "label": label_map.get(predicted_idx, "Unknown"),
            "image": base64_img
        }

    except Exception as e:
        raise ValueError(f"Image classification failed: {e}")


