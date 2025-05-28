# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application directory (app folder) into the container at /app/app
# This assumes your 'app' directory with app.py, document_processor.py, etc. is in the same directory as this Dockerfile
COPY ./app /app/app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models, uploads, and ChromaDB
# These directories will be used for volume mounting from the host
RUN mkdir -p /app/models /app/uploads /app/chroma_db

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
# for directories like /app/models, /app/uploads, /app/chroma_db if they need to be writable by the app.
# For simplicity in this exercise, we'll continue as root (default).
# The CMD would remain the same as it's executed by the current user.
# The WORKDIR /app also applies to the user.
