# app/document_processor.py
import os
from typing import List, Any
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from docx import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Option 1 for chunking

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "internal_documents"

# --- Manual Text Chunker (Option 2 - if not using Langchain) ---
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += (chunk_size - chunk_overlap)
    if not chunks and text: # Ensure at least one chunk if there is text
        chunks.append(text)
    return chunks

# --- ChromaDB Client and Collection Setup ---
# Ensure the ChromaDB directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Use a persistent client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Using sentence-transformers for embeddings
try:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
except Exception as e:
    print(f"Error initializing SentenceTransformerEmbeddingFunction: {e}")
    print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
    # Fallback to a default embedding function if SentenceTransformer fails (optional)
    # sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction() 
    raise e # Or re-raise the exception to halt if embedding is critical

# Get or create the collection
# Make sure to use the embedding function during collection creation/retrieval
try:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef, # Pass the embedding function instance
        # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric if needed
    )
except Exception as e:
    print(f"Error getting or creating ChromaDB collection '{COLLECTION_NAME}': {e}")
    raise e

# --- Document Processing Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def process_document(file_path: str, file_name: str):
    _, extension = os.path.splitext(file_name)
    text = ""
    print(f"Processing document: {file_name} from path: {file_path}")

    if extension.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif extension.lower() == ".docx":
        text = extract_text_from_docx(file_path)
    elif extension.lower() == ".txt": # Added support for .txt files
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
    else:
        print(f"Unsupported file type: {extension} for file: {file_name}")
        return

    if not text.strip(): # Check if text is empty or just whitespace
        print(f"No text extracted from {file_name} or text is empty.")
        return

    # Chunk the text (Option 2: Manual - comment out if using Langchain)
    chunks = chunk_text(text)
    # Alternative: Langchain's RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # chunks = text_splitter.split_text(text)

    if not chunks:
        print(f"No chunks created for {file_name}. This might happen if the document was empty.")
        return

    # Generate IDs and metadatas for ChromaDB
    ids = [f"{file_name}_chunk_{i}" for i, _ in enumerate(chunks)]
    metadatas = [{"source": file_name, "chunk_index": i, "original_filename": file_name} for i, _ in enumerate(chunks)]
    
    # Check for duplicates before adding
    try:
        existing_ids_response = collection.get(ids=ids) # Only get relevant IDs
        existing_ids = set(existing_ids_response['ids'])
    except Exception as e:
        print(f"Error checking existing IDs in ChromaDB: {e}")
        # Depending on desired behavior, may return or try to proceed
        return 
        
    new_chunks = []
    new_ids = []
    new_metadatas = []

    for i, chunk_id in enumerate(ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunks[i])
            new_ids.append(ids[i])
            new_metadatas.append(metadatas[i])
    
    if not new_chunks:
        print(f"All {len(chunks)} chunks from {file_name} already exist in the database.")
        return

    try:
        collection.add(
            documents=new_chunks,
            metadatas=new_metadatas,
            ids=new_ids
        )
        print(f"Successfully processed and added {len(new_chunks)} new chunks from {file_name} to ChromaDB.")
    except Exception as e:
        print(f"Error adding document {file_name} to ChromaDB: {e}")
