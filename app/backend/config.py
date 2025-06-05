from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

GENERATOR_MODEL_PATH = os.getenv("GENERATOR_MODEL_PATH")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH")
