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
