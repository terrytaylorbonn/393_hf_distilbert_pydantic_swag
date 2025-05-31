# Use a more stable and patched slim variant
FROM python:3.10-slim-bullseye

# Environment variables to reduce Python noise
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade OS packages and install essentials
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers pydantic

# Add this before loading the pipeline
RUN pip install torch

# (Optional) preload models for faster container start
RUN python -c "\
from transformers import pipeline; \
pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english'); \
pipeline('summarization', model='facebook/bart-large-cnn')"

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
