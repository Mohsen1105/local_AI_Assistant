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

### a. Download an LLM Model
The application requires a GGUF-format Large Language Model.
- Create a directory named `models` in the root of the project:
  ```bash
  mkdir models
  ```
- Download a GGUF model file and place it into the `./models` directory. For example, to use the default Mistral 7B model:
  - Go to [TheBloke/Mistral-7B-Instruct-v0.1-GGUF on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF).
  - Download a GGUF file, e.g., `mistral-7b-instruct-v0.1.Q4_K_M.gguf`.
  - Place it in `./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`.

### b. (Important) Uncomment LLM Loading Code
The application is initially configured to use a placeholder LLM to ensure it can start without a model. To use a real LLM:
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

From the root of the project directory, run:
```bash
docker build -t ai-assistant-app .
```

## 5. Run the Docker Container

```bash
docker run -p 8000:8000 \
    -v ./models:/app/models \
    -v ./chroma_db:/app/chroma_db \
    -v ./uploads:/app/uploads \
    -e ACTIVE_LLM_MODEL_FILENAME="your_model_filename.gguf" \
    # Optional: -e ACTIVE_LLM_MODEL_TYPE="mistral" \
    # Optional for NVIDIA GPU: --gpus all \
    ai-assistant-app
```
- Replace `"your_model_filename.gguf"` with the actual filename of your chosen model if it's different from the default.
- **Volumes**:
  - `./models:/app/models`: Mounts your local models directory into the container.
  - `./chroma_db:/app/chroma_db`: Persists the Chroma vector database on your host machine.
  - `./uploads:/app/uploads`: Persists uploaded files temporarily (less critical if files are processed and then potentially removed or archived by a different process).
- **Environment Variables (`-e`)**:
  - Use `-e ACTIVE_LLM_MODEL_FILENAME="your_model.gguf"` to specify a different model.
  - Use `-e ACTIVE_LLM_MODEL_TYPE="model_type"` if you need to specify the model type for `ctransformers`.
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

The application includes Python functions in `app/data_connectors.py` to connect to SQL Server and Oracle databases. However, using these functions requires:
1.  **Database Drivers**: The necessary ODBC drivers (for SQL Server) or Oracle Instant Client libraries must be installed *within the Docker container*. This requires modifying the `Dockerfile` to include the installation steps for these system-level dependencies, which can be complex and specific to your OS/environment.
2.  **Python Packages**: `pyodbc` and `cx_Oracle` are listed in `requirements.txt`. They might require compilation against the installed drivers/client libraries.
3.  **Connection Details**: You'll need to provide valid connection strings/credentials when calling the respective functions (currently not exposed via API endpoints directly for querying, but can be used internally or via new endpoints if developed).

**This is an advanced setup step. The base Dockerfile does not include these database drivers.**

## 8. Stopping the Application
- To stop the running Docker container, find its ID using `docker ps` and then run `docker stop <container_id>`.

```
