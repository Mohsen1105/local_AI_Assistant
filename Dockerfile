# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install supervisor and create log directory
RUN apt-get update && \
    apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/log/supervisor

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
COPY ${HOST_SENTENCE_TRANSFORMER_MODEL_PATH} /app/sentence_transformer_models/all-MiniLM-L6-v2/

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application directory (app folder) into the container at /app/app
COPY ./app /app/app

# Copy the Streamlit app file into the container at /app
COPY streamlit_app.py /app/streamlit_app.py

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Make ports available to the world outside this container
EXPOSE 8000 # For FastAPI
EXPOSE 8501 # For Streamlit

# Define the command to run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# Optional: Add a non-root user (good practice)
# RUN useradd --create-home appuser
# USER appuser
# Note: If using a non-root user, ensure file permissions are correctly set
# for directories like /app/models, /app/uploads, /app/chroma_db and the sentence_transformer_models
# if they need to be writable by the app (though sentence transformer models are usually read-only at runtime).
# Also ensure the supervisor log directory is writable by this user, or configure supervisor to log elsewhere.
# For simplicity in this exercise, we'll continue as root (default).
# The CMD would remain the same as it's executed by the current user.
# The WORKDIR /app also applies to the user.
