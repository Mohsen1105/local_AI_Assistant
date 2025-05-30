# This is the main application file for the FastAPI LLM server.

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
from contextlib import asynccontextmanager
import os
import shutil
from typing import Dict, List, Optional # Added Optional for clarity, though not strictly needed by the code here

from . import document_processor # Assuming document_processor.py is in the same 'app' directory

import chromadb
from chromadb.utils import embedding_functions


# --- Authentication & Authorization Configuration ---
API_KEYS_ROLES: Dict[str, List[str]] = {
    "supersecretadminapikey": ["admin", "query_user", "ingestion_user", "correspondence_user"], # Admin has all roles
    "user123queryapikey": ["query_user"],
    "datauploader456apikey": ["ingestion_user"],
    "consultant789apikey": ["query_user", "correspondence_user"] 
}

# Define roles
ROLE_ADMIN = "admin"
ROLE_QUERY_USER = "query_user"
ROLE_INGESTION_USER = "ingestion_user"
ROLE_CORRESPONDENCE_USER = "correspondence_user"

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # auto_error=False to handle custom exception

async def get_api_key_roles(key: str = Security(api_key_header)) -> List[str]:
    if not key: # Check if key is None or empty, which happens if header is missing
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key header missing",
        )
    if key in API_KEYS_ROLES:
        print(f"API Key validated. Roles: {API_KEYS_ROLES[key]}")
        return API_KEYS_ROLES[key]
    else:
        print(f"Invalid API Key received: {key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# --- Role Specific Dependencies ---
def create_role_dependency(role_name: str):
    async def role_checker(roles: List[str] = Security(get_api_key_roles)):
        if role_name not in roles:
            print(f"Access denied for role '{role_name}'. User roles: {roles}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have the required '{role_name}' role"
            )
        print(f"Access granted for role '{role_name}'. User roles: {roles}")
        return roles # Or just True, or the specific role
    return role_checker

# Create dependencies for each role
require_admin = create_role_dependency(ROLE_ADMIN)
require_query = create_role_dependency(ROLE_QUERY_USER)
require_ingestion = create_role_dependency(ROLE_INGESTION_USER)
require_correspondence = create_role_dependency(ROLE_CORRESPONDENCE_USER)

# --- Main Application Configuration & Lifespan ---

# --- LLM Configuration ---
# Model filename (e.g., "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
# This file is expected to be in the ./models directory
DEFAULT_MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf" # A sensible default
ACTIVE_LLM_MODEL_FILENAME = os.environ.get("ACTIVE_LLM_MODEL_FILENAME", DEFAULT_MODEL_FILENAME)

# Model type (e.g., "mistral", "llama", "phi"). 
# ctransformers often infers this, but it can be specified if needed.
ACTIVE_LLM_MODEL_TYPE = os.environ.get("ACTIVE_LLM_MODEL_TYPE", None) 

MODELS_DIR = "./models" # Directory where models are stored
MODEL_PATH_CONFIG = os.path.join(MODELS_DIR, ACTIVE_LLM_MODEL_FILENAME)
MODEL_REPO_ID_INFO = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF (example)" # For user info

print(f"INFO: Attempting to use model: {MODEL_PATH_CONFIG}")
if ACTIVE_LLM_MODEL_TYPE:
    print(f"INFO: Using specified model type: {ACTIVE_LLM_MODEL_TYPE}")
else:
    print("INFO: Model type not specified, ctransformers will attempt auto-detection if/when loaded.")


# --- General App Configuration ---
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHROMA_DB_PATH = "./chroma_db" # Used by document_processor and now query
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


llm = None # Global LLM instance, to be populated by lifespan manager

# --- ChromaDB Access for Querying (Initialized once on startup) ---
CHROMA_DB_PATH_QUERY = CHROMA_DB_PATH # Same path as ingestion
EMBEDDING_MODEL_NAME_QUERY = "all-MiniLM-L6-v2" # Same model as ingestion
COLLECTION_NAME_QUERY = "internal_documents" # Same collection as ingestion

query_client = None
query_sentence_transformer_ef = None
query_collection = None

try:
    # Initialize client and embedding function
    print(f"Initializing ChromaDB client for querying at path: {CHROMA_DB_PATH_QUERY}")
    query_client = chromadb.PersistentClient(path=CHROMA_DB_PATH_QUERY)
    
    print(f"Initializing SentenceTransformerEmbeddingFunction with model: {EMBEDDING_MODEL_NAME_QUERY}")
    query_sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME_QUERY
    )
    
    # Get the existing collection
    print(f"Attempting to get ChromaDB collection: '{COLLECTION_NAME_QUERY}'")
    query_collection = query_client.get_collection(
        name=COLLECTION_NAME_QUERY,
        embedding_function=query_sentence_transformer_ef # Crucial: pass the embedding function
    )
    print(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME_QUERY}' for querying.")
except Exception as e:
    # Detailed error logging is important here
    print(f"CRITICAL ERROR: Failed to connect to or get ChromaDB collection '{COLLECTION_NAME_QUERY}' for querying.")
    print(f"Error details: {e}")
    print("The '/query-documents/' endpoint will not function correctly.")
    # query_collection will remain None. The endpoint should check for this.

# --- Lifespan Management (Model Loading and Unloading) ---
# --- Placeholder LLM Classes (can be moved to a separate file if they grow) ---
class DummyLLM:
    def __init__(self, model_path="N/A"):
        self.model_path = model_path
        print(f"DummyLLM initialized. (Intended model: {self.model_path})")

    def __call__(self, prompt, **kwargs):
        # Adjusted to be slightly more context-aware for different tasks
        if "summarize" in prompt.lower():
            text_to_summarize_marker = "Text to summarize:"
            text_start_index = prompt.find(text_to_summarize_marker)
            if text_start_index != -1:
                actual_text_start = text_start_index + len(text_to_summarize_marker)
                text_snippet = prompt[actual_text_start:].strip()[:70] # Get first 70 chars of actual text
                return f"Simulated summary of text starting with: '{text_snippet}...' (max_length hint: {kwargs.get('max_summary_length', 'N/A')}, model: {self.model_path})"
            return f"Simulated summary of: '{prompt[:100]}...' (model: {self.model_path})"
        # Fallback for other prompts (like RAG or direct generation)
        return f"Simulated LLM response for prompt starting with: '{prompt[:150]}...' (model: {self.model_path})"

class MissingLLM:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"MissingLLM initialized. Model not found at: {self.model_path}")

    def __call__(self, prompt, **kwargs):
        raise RuntimeError(f"Model not loaded. Please ensure '{self.model_path}' exists and is correctly configured.")

class ErrorLLM:
    def __init__(self, model_path, error_message):
        self.model_path = model_path
        self.error_message = error_message
        print(f"ErrorLLM initialized. Error loading model {self.model_path}: {self.error_message}")

    def __call__(self, prompt, **kwargs):
        raise RuntimeError(f"Model '{self.model_path}' failed to load due to: {self.error_message}")

# --- Lifespan Management (Model Loading and Unloading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    try:
        print(f"Lifespan: Attempting to load model from: {MODEL_PATH_CONFIG}")
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            print(f"Lifespan: '{MODELS_DIR}' directory created.")
        
        if not os.path.exists(MODEL_PATH_CONFIG):
            print(f"Lifespan: Model file not found at {MODEL_PATH_CONFIG}. Please ensure it's downloaded.")
            llm = MissingLLM(model_path=MODEL_PATH_CONFIG) 
        else:
            print("Lifespan: Model file found. Actual ctransformers loading is commented out for now.")
            # IMPORTANT: The following ctransformers line should be uncommented by the user 
            # when they have a model and are ready to test actual LLM inference.
            # from ctransformers import AutoModelForCausalLM # Keep import here for when it's uncommented
            # llm = AutoModelForCausalLM.from_pretrained(
            #     MODEL_PATH_CONFIG,
            #     model_type=ACTIVE_LLM_MODEL_TYPE if ACTIVE_LLM_MODEL_TYPE else None,
            #     # Add other ctransformers parameters if needed, e.g., gpu_layers=50 for GPU offload
            #     # Example: context_length=4096
            # )
            # print(f"Lifespan: Successfully loaded LLM: {ACTIVE_LLM_MODEL_FILENAME} of type {ACTIVE_LLM_MODEL_TYPE or 'auto-detected'}")
            
            # For now, to ensure the app runs without ctransformers fully installed or model downloaded by default:
            if 'llm' not in globals() or llm is None: # Keep dummy if actual load is commented
                llm = DummyLLM(model_path=MODEL_PATH_CONFIG) 
                print(f"Lifespan: Using DummyLLM for {ACTIVE_LLM_MODEL_FILENAME}.")

    except Exception as e:
        error_msg = str(e)
        print(f"Lifespan: Error loading LLM model {ACTIVE_LLM_MODEL_FILENAME}: {error_msg}")
        llm = ErrorLLM(model_path=MODEL_PATH_CONFIG, error_message=error_msg)
    yield
    # Clean up the ML models and release the resources
    print("Shutting down application and cleaning up resources.")
    llm = None # Release the model object

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/generate")
async def generate_text(payload: dict = Body(...)):
    """
    Generates text using the loaded LLM based on the provided prompt.
    Expects a JSON payload with a "prompt" field.
    Example: {"prompt": "Translate 'hello' to French."}
    """
    global llm
    if llm is None:
        raise HTTPException(status_code=503, detail="Language model not loaded. Please check server logs.")
    
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request payload.")

    try:
        print(f"Received prompt: {prompt}")
        # Note: Actual generation parameters (max_tokens, temperature, etc.) can be added here.
        # For ctransformers, the __call__ method of the model object performs generation.
        generated_text = llm(prompt, max_new_tokens=150, temperature=0.7) 
        print(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}
    except RuntimeError as e: # Catch errors from our dummy/missing LLM handlers
        print(f"Runtime error during text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate text.")

@app.get("/")
async def read_root():
    return {"message": "LLM Inference Server is running. POST to /generate with a 'prompt'."}

# --- Pydantic Models for API Requests ---
class QueryRequest(BaseModel):
    """
    Request model for querying documents.
    Requires a query string and an optional number of top matching document chunks to retrieve.
    """
    query: str
    top_n: int = 3 # Number of relevant chunks to retrieve

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What were the main operational issues reported last month?",
                "top_n": 5,
            }
        }

class SummarizeRequest(BaseModel):
    """
    Request model for text summarization.
    Requires the text to be summarized and an optional maximum length for the summary.
    """
    text: str
    max_summary_length: int = 150 # Optional: hint for summary length

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The quick brown fox jumps over the lazy dog. This sentence is often used to demonstrate all letters of the alphabet. It is a classic pangram. The purpose of this example is to provide a long enough text that might warrant summarization into a shorter form, perhaps focusing on the key subject, which is the pangram itself and its usage.",
                "max_summary_length": 50,
            }
        }

class FreeFormPromptRequest(BaseModel):
    """
    Request model for free-form interaction with the LLM.
    Requires a prompt string.
    """
    prompt: str
    # max_tokens: int = 500 # Example if we want to control LLM output length later

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain the concept of a Large Language Model in simple terms.",
            }
        }

# --- Query Endpoint ---
@app.post("/query-documents/", 
            summary="Query internal documents (RAG)",
            dependencies=[Security(require_query)])
async def query_documents_endpoint(request: QueryRequest): # Renamed to avoid conflict with module
    """
    Accepts a user query, retrieves relevant document chunks from the vector store (ChromaDB),
    constructs a context-augmented prompt, and uses the LLM to generate an answer.

    This endpoint implements the Retrieval Augmented Generation (RAG) pattern.
    Requires 'query_user' role.
    """
    global llm # Access the global llm instance (populated by lifespan)
    global query_collection # Access the global query_collection instance

    if query_collection is None:
        # This check is critical if ChromaDB initialization failed
        print("Error: query_collection is None. ChromaDB might not be initialized correctly.")
        raise HTTPException(status_code=503, detail="ChromaDB collection not available. Check server logs for connection errors during startup.")

    if not request.query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request payload.")

    try:
        print(f"Received query: '{request.query}', top_n: {request.top_n}")

        # Search ChromaDB
        # The query_collection's embedding function will automatically handle embedding the query_texts
        results = query_collection.query(
            query_texts=[request.query],
            n_results=request.top_n,
            include=['documents', 'metadatas'] # Ensure these are included
        )
        
        retrieved_documents = results.get('documents', [[]])[0] # Safely get documents, default to empty list
        retrieved_metadatas = results.get('metadatas', [[]])[0] # Safely get metadatas

        if not retrieved_documents:
            print(f"No relevant documents found in ChromaDB for the query: '{request.query}'")
            # You can choose to return a message here or let the LLM respond based on no context
            # For now, we'll proceed and let the LLM know no context was found.

        # Format context for LLM
        context_str = "\n\n---\n\n".join(retrieved_documents) if retrieved_documents else "No relevant context found in documents."

        # Construct a prompt for the LLM
        llm_prompt = f"""Based on the following context, please answer the query.
Context:
{context_str}

Query: {request.query}
Answer:"""
        
        print(f"Constructed LLM prompt (first 300 chars): {llm_prompt[:300]}...")

        # Call the LLM (DummyLLM or actual model via global llm instance)
        if llm:
            try:
                # The DummyLLM __call__ takes prompt and **kwargs, so this is fine.
                # Actual ctransformer model also has a __call__ method.
                generated_text = llm(llm_prompt) 
            except RuntimeError as e: # Catch errors from our dummy/missing/error LLM handlers
                print(f"Runtime error from LLM during query: {e}")
                raise HTTPException(status_code=500, detail=f"LLM runtime error: {str(e)}")
            except Exception as e_llm:
                print(f"Unexpected error from LLM during query: {e_llm}")
                raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e_llm)}")
        else:
            # This case should ideally be caught by lifespan if model loading fails initially,
            # or if llm is explicitly set to None for some reason post-startup.
            print("CRITICAL: LLM instance is None at query time. This should have been handled by lifespan.")
            raise HTTPException(status_code=503, detail="LLM not available. Check server logs for loading errors.")

        return {
            "query": request.query,
            "answer": generated_text,
            "retrieved_context_count": len(retrieved_documents),
            "retrieved_context_metadatas": retrieved_metadatas
        }

    except HTTPException as he: # Re-raise HTTPExceptions directly to preserve status code and details
        raise he
    except Exception as e:
        # Catch-all for other unexpected errors during the query process
        print(f"Error during document query processing: {e}")
        # Log the exception e (e.g., using a proper logger)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# --- Summarization Endpoint ---
@app.post("/summarize-text/", 
            summary="Summarize provided text",
            dependencies=[Security(require_query)]) # Protected with query role
async def summarize_text_endpoint(request: SummarizeRequest):
    """
    Accepts a block of text and generates a concise summary using the LLM.
    The desired maximum length of the summary can be specified.

    Requires 'query_user' role (as it uses LLM resources, similar to querying).
    """
    global llm # Access the global llm instance

    if not request.text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request payload for summarization.")

    if llm is None:
        # This should ideally be caught by lifespan if model loading fails catastrophically,
        # but good to double check.
        print("CRITICAL: LLM instance is None at summarization time.")
        raise HTTPException(status_code=503, detail="LLM not available. Check server logs for loading errors.")

    try:
        print(f"Received text for summarization (first 100 chars): '{request.text[:100]}...', max_length: {request.max_summary_length}")

        llm_prompt = f"""Please summarize the following text concisely. Aim for a summary of around {request.max_summary_length} words if possible, but prioritize clarity and key information:

Text to summarize:
{request.text}

Summary:"""
        
        print(f"Constructed summarization prompt (first 200 chars): {llm_prompt[:200]}...")

        # Call the LLM (DummyLLM or actual model)
        # The DummyLLM __call__ can take **kwargs, so we pass max_summary_length for it to use.
        generated_summary = llm(llm_prompt, max_summary_length=request.max_summary_length) 
        
        print(f"Generated summary (simulated): {generated_summary}")

        return {
            "original_text": request.text,
            "summary": generated_summary,
        }
    except RuntimeError as e: # Catch errors from our dummy/missing/error LLM handlers
        print(f"Runtime error from LLM during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"LLM runtime error: {str(e)}")
    except HTTPException as he: # Re-raise HTTPExceptions directly
        raise he
    except Exception as e:
        print(f"Error during text summarization: {e}")
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error processing summarization request: {str(e)}")

# --- Free-form Prompt / Consultation Endpoint ---
@app.post("/generate-response/", 
            summary="Generate response from free-form prompt (Consultation)",
            dependencies=[Security(require_correspondence)])
async def generate_response_endpoint(request: FreeFormPromptRequest):
    """
    Accepts a free-form prompt from the user and returns a direct response from the LLM.
    This endpoint is suitable for general consultation, correspondence generation,
    or any task that doesn't require specific document retrieval (RAG).

    Requires 'correspondence_user' role.
    """
    global llm # Access the global llm instance

    if not request.prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request payload.")

    if llm is None:
        print("CRITICAL: LLM instance is None at generate-response time.")
        raise HTTPException(status_code=503, detail="LLM not available. Check server logs for loading errors.")

    try:
        print(f"Received free-form prompt (first 100 chars): '{request.prompt[:100]}...'")

        # Using the user's prompt directly for now.
        # Could add framing here if needed, e.g., for specific consultation types.
        llm_api_prompt = request.prompt
        
        # The DummyLLM will provide a generic simulated response.
        # If specific parameters like max_tokens were defined in FreeFormPromptRequest,
        # they could be passed here, e.g., llm(llm_api_prompt, max_tokens=request.max_tokens)
        generated_text = llm(llm_api_prompt) 
        
        print(f"Generated response (simulated): {generated_text}")

        return {
            "user_prompt": request.prompt,
            "generated_response": generated_text,
        }
    except RuntimeError as e: # Catch errors from our dummy/missing/error LLM handlers
        print(f"Runtime error from LLM during free-form generation: {e}")
        raise HTTPException(status_code=500, detail=f"LLM runtime error: {str(e)}")
    except HTTPException as he: # Re-raise HTTPExceptions directly
        raise he
    except Exception as e:
        print(f"Error during free-form response generation: {e}")
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error processing free-form request: {str(e)}")


@app.post("/upload-document/", 
            summary="Upload document for ingestion",
            dependencies=[Security(require_ingestion)])
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (PDF, DOCX, TXT) to the server.
    The server then processes the document, extracts text, chunks it,
    generates embeddings, and ingests them into the vector database (ChromaDB)
    for later retrieval.

    Requires 'ingestion_user' role.
    """
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        print(f"Attempting to save uploaded file to: {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved successfully to {temp_file_path}")
        
        # Call the document processor
        print(f"Calling document_processor for {file.filename}")
        document_processor.process_document(temp_file_path, file.filename)
        print(f"Document processing finished for {file.filename}")
        
        return {"message": f"Document '{file.filename}' processed and ingested successfully."}
    except HTTPException as he: # Re-raise HTTPExceptions directly
        raise he
    except Exception as e:
        print(f"Error processing document {file.filename}: {e}")
        # Log the exception e (e.g., using a proper logger)
        raise HTTPException(status_code=500, detail=f"Error processing document '{file.filename}': {str(e)}")
    finally:
        # Ensure the temporary file is removed after processing
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Temporary file {temp_file_path} removed.")
            except Exception as e_remove:
                print(f"Error removing temporary file {temp_file_path}: {e_remove}")
        # Clean up the UploadFile's temp file if it's still open
        # (FastAPI usually handles this, but good for robustness)
        await file.close()
        print(f"Closed file stream for {file.filename}")


# --- How to Run ---
# To run this FastAPI application:
# 1. Make sure you have all dependencies installed (from requirements.txt):
#    pip install -r requirements.txt
# 2. (Model Download - separate step) Ensure the GGUF model file (e.g., mistral-7b-instruct-v0.1.Q4_K_M.gguf)
#    is downloaded to the './models' directory or the path specified by the MODEL_PATH environment variable.
#    For example, from Hugging Face Hub:
#    `huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False`
# 3. Run Uvicorn:
#    uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
#
#    Or if app.py is in the root:
#    uvicorn app:app --reload --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    # This part is for debugging and direct execution,
    # but uvicorn is the standard way to run FastAPI apps.
    import uvicorn
    print("Starting Uvicorn server directly (for debugging)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
