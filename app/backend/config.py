from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), ".env")
# Load environment variables from .env
load_dotenv(env_path)
GENERATOR_MODEL_PATH = os.getenv("GENERATOR_MODEL_PATH")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH")
