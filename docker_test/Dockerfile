# Base image with PyTorch + CUDA 11.8
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Copy a simple test script (create this file locally)
COPY test.py .

# Run the script when the container starts
CMD ["python", "test.py"]
