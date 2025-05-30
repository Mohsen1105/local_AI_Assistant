# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Argument for the host path where the sentence transformer model files are located
# Users should override this during build: --build-arg HOST_SENTENCE_TRANSFORMER_MODEL_PATH=./path/to/your/model
# Using a default that might exist locally if users follow a convention.
ARG HOST_SENTENCE_TRANSFORMER_MODEL_PATH=./local_sentence_transformer_models/all-MiniLM-L6-v2

# Create directories for LLM models, uploads, ChromaDB, and the sentence transformer model target path
# The target path /app/sentence_transformer_models/all-MiniLM-L6-v2 should match the default in app.py and document_processor.py
RUN mkdir -p /app/models /app/uploads /app/chroma_db /app/sentence_transformer_models/all-MiniLM-L6-v2

# Copy the pre-downloaded sentence transformer model into the image
# Ensure HOST_SENTENCE_TRANSFORMER_MODEL_PATH is set during docker build via --build-arg
# If HOST_SENTENCE_TRANSFORMER_MODEL_PATH is not set or points to a non-existent/empty directory,
# this COPY command might behave unexpectedly or copy nothing, so it's crucial the user provides a valid path.
# The trailing slash on the destination ensures the contents of the source directory are copied into the target directory.
COPY ${HOST_SENTENCE_TRANSFORMER_MODEL_PATH} /app/sentence_transformer_models/all-MiniLM-L6-v2/

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application directory (app folder) into the container at /app/app
# This assumes your 'app' directory with app.py, document_processor.py, etc. is in the same directory as this Dockerfile
COPY ./app /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application using Uvicorn
# This tells Docker how to run your FastAPI application
# It assumes your FastAPI 'app' instance is in /app/app/app.py (app.app:app)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Optional: Add a non-root user (good practice)
# RUN useradd --create-home appuser
# USER appuser
# Note: If using a non-root user, ensure file permissions are correctly set 
# for directories like /app/models, /app/uploads, /app/chroma_db and the sentence_transformer_models
# if they need to be writable by the app (though sentence transformer models are usually read-only at runtime).
# For simplicity in this exercise, we'll continue as root (default).
# The CMD would remain the same as it's executed by the current user.
# The WORKDIR /app also applies to the user.
