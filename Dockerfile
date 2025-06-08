# Use official Python image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime


# Avoid writing pyc files and force output flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy the entire project directory into /app
COPY . /app

# Move into the backend folder
WORKDIR /app/app/backend

# Install backend dependencies
RUN pip install -r requirements.txt

# Install R3GAN as an editable local package
RUN pip install -e ../../

# Expose the backend port (adjust if needed)
EXPOSE 8000

# Command to run your FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
