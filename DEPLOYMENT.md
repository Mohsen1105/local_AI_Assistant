# AI Assistant Deployment Guide

This guide provides instructions on how to build and run the AI Assistant application using Docker.

## Prerequisites

- Docker installed on your system.
- Git installed (for cloning the repository).
- At least 15-20GB of free disk space for Docker images, models, and data.
- Sufficient RAM (ideally 16GB+, depending on the LLM used).
- An NVIDIA GPU with CUDA support is recommended for LLM acceleration (requires NVIDIA Docker runtime).

## 1. Clone the Repository

```bash
# git clone <repository_url>
# cd <repository_directory>
```
*(Replace `<repository_url>` and `<repository_directory>` with actual values if known, otherwise use placeholders)*

## 2. Initial Setup

### a. Download Required Models

#### i. Download GGUF LLM Model (for main text generation)
The application requires a GGUF-format Large Language Model for its primary text generation capabilities.
- Create a directory named `models` in the root of the project:
  ```bash
  mkdir models
  ```
- Download a GGUF model file and place it into the `./models` directory. For example, to use the default Mistral 7B model:
  - Go to [TheBloke/Mistral-7B-Instruct-v0.1-GGUF on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).
  - Download a GGUF file, e.g., `mistral-7b-instruct-v0.1.Q4_K_M.gguf`.
  - Place it in `./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`.

#### ii. Download Sentence Transformer Model (Mandatory for Airgapped/Self-Contained Build)

The application uses the `sentence-transformers/all-MiniLM-L6-v2` model for generating embeddings for document retrieval. To build a self-contained Docker image that includes this model (recommended for airgapped environments or simplified deployment):

1.  **Download the model files:**
    You need to download the complete model repository. The easiest way is often to use `git clone` if you have Git LFS installed, or download the files manually from the Hugging Face Hub:
    - Go to [sentence-transformers/all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
    - Click on the "Files and versions" tab.
    - Download all files and place them into a local directory. For example, create `local_sentence_transformer_models/all-MiniLM-L6-v2` in your project root and place all model files (config.json, pytorch_model.bin, vocab.txt, special_tokens_map.json, etc.) directly in that folder. *Ensure the path corresponds to what you will pass during the Docker build.*

2.  **Structure:**
    Example local structure:
    ```
    your_project_root/
    ├── local_sentence_transformer_models/
    │   └── all-MiniLM-L6-v2/    <-- Place all model files directly here
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       ├── tokenizer_config.json
    │       └── ... (all other files from the HF repo)
    ├── app/
    ├── models/ (for GGUF LLMs)
    ├── Dockerfile
    └── ...
    ```

### c. (Important) Uncomment LLM Loading Code
The application is initially configured to use a placeholder LLM to ensure it can start without a model. To use a real LLM for text generation:
- Open `app/app.py`.
- Find the `lifespan` function.
- **Comment out** the line that initializes the `DummyLLM` or `MissingLLM` (e.g., `llm = DummyLLM(...)` or `llm = MissingLLM(...)`).
- **Uncomment** the lines responsible for loading the actual model with `ctransformers`:
  ```python
  # from ctransformers import AutoModelForCausalLM # Ensure this import is active
  # llm = AutoModelForCausalLM.from_pretrained(
  #     MODEL_PATH_CONFIG,
  #     model_type=ACTIVE_LLM_MODEL_TYPE if ACTIVE_LLM_MODEL_TYPE else None,
  #     # For GPU offload (example: 50 layers), uncomment and adjust if you have CUDA:
  #     # gpu_layers=50 
  # )
  # print(f"Lifespan: Successfully loaded LLM: {ACTIVE_LLM_MODEL_FILENAME}")
  ```
- Ensure the `from ctransformers import AutoModelForCausalLM` line is also active/uncommented near the top of the file where other imports are.

## 3. Configure API Keys & LLM (Optional)

### a. API Keys
API keys and their associated roles are currently managed in a dictionary within `app/app.py` (the `API_KEYS_ROLES` variable). You can modify this dictionary directly to add, remove, or change API keys and roles before building the Docker image if needed for your local setup.

### b. LLM Model Selection
You can switch the LLM model by setting environment variables when running the Docker container (see Step 5). The default is `mistral-7b-instruct-v0.1.Q4_K_M.gguf`.
- `ACTIVE_LLM_MODEL_FILENAME`: Name of the GGUF file in the `./models` directory.
- `ACTIVE_LLM_MODEL_TYPE`: (Optional) The model type (e.g., `mistral`, `llama`). `ctransformers` often auto-detects this.

## 4. Build the Docker Image

To build the Docker image, which now includes the Sentence Transformer model, run the following command from the root of the project directory. You **must** provide the path to your locally downloaded Sentence Transformer model files using the `HOST_SENTENCE_TRANSFORMER_MODEL_PATH` build argument.

```bash
docker build --build-arg HOST_SENTENCE_TRANSFORMER_MODEL_PATH="./local_sentence_transformer_models/all-MiniLM-L6-v2" -t ai-assistant-app .
```
- Replace `./local_sentence_transformer_models/all-MiniLM-L6-v2` with the actual path to your downloaded `all-MiniLM-L6-v2` model files if you stored them elsewhere.
- If you used a different directory name than `all-MiniLM-L6-v2` inside your host path, ensure the `COPY` command in the `Dockerfile` and the default path in the application code (`ENV_SENTENCE_TRANSFORMER_MODEL_PATH`) are consistent, or use the environment variable to point to the correct internal path. The default internal path is `/app/sentence_transformer_models/all-MiniLM-L6-v2`.

## 5. Run the Docker Container

```bash
docker run -p 8000:8000 \
    -v ./models:/app/models \
    -v ./chroma_db:/app/chroma_db \
    -v ./uploads:/app/uploads \
    -e ACTIVE_LLM_MODEL_FILENAME="your_model_filename.gguf" \
    # Optional: -e ACTIVE_LLM_MODEL_TYPE="mistral" \
    # Optional: -e ENV_SENTENCE_TRANSFORMER_MODEL_PATH="/app/custom_st_model_path" \
    # Optional for NVIDIA GPU: --gpus all \
    ai-assistant-app
```
- Replace `"your_model_filename.gguf"` with the actual filename of your chosen GGUF LLM if it's different from the default.
- **Volumes**:
  - `./models:/app/models`: Mounts your local GGUF LLM models directory into the container.
  - `./chroma_db:/app/chroma_db`: Persists the Chroma vector database on your host machine.
  - `./uploads:/app/uploads`: Persists uploaded files temporarily.
- **Environment Variables (`-e`)**:
  - `ACTIVE_LLM_MODEL_FILENAME`: Name of the GGUF file in the `./models` directory (for the main LLM).
  - `ACTIVE_LLM_MODEL_TYPE`: (Optional) The model type for the main GGUF LLM.
  - `ENV_SENTENCE_TRANSFORMER_MODEL_PATH`: (Optional) Path inside the container where the Sentence Transformer model is located. The default is `/app/sentence_transformer_models/all-MiniLM-L6-v2`, which is where the Dockerfile copies the model. You generally do not need to set this unless you customize the Dockerfile's internal paths.
- **GPU Acceleration**:
  - If you have an NVIDIA GPU and want to use it, you need `nvidia-docker2` (NVIDIA Container Toolkit) installed.
  - Add the `--gpus all` flag to the `docker run` command.
  - You might also need to install `ctransformers` with CUDA support. The current `requirements.txt` installs the CPU version. To build with GPU support, you might need to modify `requirements.txt` to `ctransformers[cuda]>=0.2.27` (or a specific version) and potentially adjust the Dockerfile if system CUDA libraries are needed. *This is an advanced step.*

## 6. Accessing the API

- **API Endpoints**: Available at `http://localhost:8000`.
- **Swagger UI (Interactive Docs)**: `http://localhost:8000/docs`
- **ReDoc (Alternative Docs)**: `http://localhost:8000/redoc`

You will need to include your API key in the `X-API-KEY` header for protected endpoints.

## 7. Database Connectivity (SQL Server / Oracle)

The application includes Python functions in `app/data_connectors.py` to connect to SQL Server and Oracle databases. However, these are **optional features** and the necessary Python packages (`pyodbc`, `cx_Oracle`) and system-level database drivers are **not installed by default** with the main `requirements.txt` or in the base Docker image.

The functions in `app/data_connectors.py` will print a warning if the required Python libraries are not found and will raise a `RuntimeError` if you attempt to use the database connectivity functions without them.

### Installing Optional Database Drivers (Advanced)

If you require connectivity to SQL Server or Oracle databases, you will need to take additional steps, as the necessary Python packages and system-level drivers are **not included** in the default Docker build to keep the image lightweight and avoid build issues for users who don't need these databases.

1.  **Install Python Packages**:
    The Python packages `pyodbc` (for SQL Server) and `cx_Oracle` (for Oracle) are listed in `requirements-db.txt`. To install them into your environment (ideally a virtual environment if working locally, or when customizing the Dockerfile):
    ```bash
    pip install -r requirements-db.txt
    ```

2.  **Install System-Level Drivers (Crucial for Docker):**
    -   **For SQL Server (`pyodbc`):** You need to install ODBC drivers within your Docker image. This typically involves adding commands to your `Dockerfile` like:
        ```dockerfile
        # Example for Debian/Ubuntu-based images - may need adjustments
        # RUN apt-get update && apt-get install -y gnupg unixodbc-dev
        # RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
        # RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
        # RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools
        ```
        Consult Microsoft's documentation for the latest ODBC driver installation instructions for your chosen base Linux distribution.
    -   **For Oracle (`cx_Oracle`):** You need to install Oracle Instant Client libraries within your Docker image. This involves downloading the Instant Client zip files from Oracle's website and setting environment variables like `LD_LIBRARY_PATH`. Example Dockerfile additions:
        ```dockerfile
        # Example - paths and versions will vary
        # ARG ORACLE_INSTANT_CLIENT_VERSION=19.10
        # ADD https://download.oracle.com/otn_software/linux/instantclient/${ORACLE_INSTANT_CLIENT_VERSION}/instantclient-basic-${ORACLE_INSTANT_CLIENT_VERSION}.0.0.0dbru.zip /tmp/
        # ADD https://download.oracle.com/otn_software/linux/instantclient/${ORACLE_INSTANT_CLIENT_VERSION}/instantclient-sdk-${ORACLE_INSTANT_CLIENT_VERSION}.0.0.0dbru.zip /tmp/
        # RUN unzip /tmp/instantclient-basic-*.zip -d /opt/oracle && \
        #     unzip /tmp/instantclient-sdk-*.zip -d /opt/oracle && \
        #     rm -rf /tmp/*.zip
        # ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_19_10:$LD_LIBRARY_PATH 
        # (Adjust version number in LD_LIBRARY_PATH)
        ```
        This requires accepting Oracle's license terms and potentially creating an Oracle account to download the client.

3.  **Rebuild Docker Image:**
    After modifying your `Dockerfile` (e.g., adding the system driver installations and potentially a `COPY requirements-db.txt /app/` and `RUN pip install -r requirements-db.txt` line), you would need to rebuild your custom Docker image.

**The base `Dockerfile` provided with this project does not include these steps to avoid complexity and large image sizes for users who do not need these specific database connections.** The functions in `app/data_connectors.py` will raise a `RuntimeError` if these optional dependencies are not installed.

## 8. Stopping the Application
- To stop the running Docker container, find its ID using `docker ps` and then run `docker stop <container_id>`.
```

```
