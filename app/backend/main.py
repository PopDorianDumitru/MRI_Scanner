from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
import os
from preprocessor.Preprocessor import Preprocessor
from api import classifier, generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_ORIENTATIONS = {"sagittal", "coronal", "axial"}
VALID_LABELS = {"healthy", "slight", "mild", "severe", "very-severe"}

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.include_router(classifier.router, prefix="/classify", tags=["Classifier"])
app.include_router(generator.router, prefix="/generate", tags=["Generator"])


# @app.post("/classify")
# async def classify_route(
#     file: UploadFile = File(...),
#     orientation: str = Form(...)
# ):
#     # Validate file
#     if not file.filename.endswith(".nii.gz"):
#         raise HTTPException(status_code=400, detail="Only .nii.gz files are accepted.")

#     # Validate orientation
#     if orientation not in VALID_ORIENTATIONS:
#         raise HTTPException(status_code=400, detail="Invalid orientation value.")

#     # Read and process
#     contents = await file.read()
#     # label = classify_mri(contents, orientation)
#     label = "Healthy"
#     return JSONResponse({"label": label})


# @app.post("/generate")
# async def generate_route(
#     label: str = Form(...),
#     count: int = Form(...)
# ):
#     if label not in VALID_LABELS:
#         raise HTTPException(status_code=400, detail="Invalid dementia label.")
#     if not (0 <= count <= 5):
#         raise HTTPException(status_code=400, detail="Count must be between 0 and 5.")

#     image_path = os.path.join(os.path.dirname(__file__), "brain_logo.png")

#     try:
#         with open(image_path, "rb") as f:
#             image_bytes = f.read()
#     except FileNotFoundError:
#         raise HTTPException(status_code=500, detail="Test image not found.")

#     image_bytes_list = [image_bytes for _ in range(count)]

#     # Return multiple images as base64, or serve individually depending on UI
#     # Here we return them as a list of base64-encoded PNGs
#     import base64
#     encoded_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
#     return JSONResponse({"images": encoded_images})