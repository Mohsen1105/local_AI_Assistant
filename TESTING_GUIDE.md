# AI Assistant Testing and Refinement Guide

This guide provides a set of test cases and areas for refinement to ensure the AI Assistant application functions correctly and provides quality results. It assumes you have followed the `DEPLOYMENT.md` to set up and run the application with a real GGUF LLM.

## 1. Prerequisites for Testing

-   The application Docker container is running.
-   You have downloaded a GGUF model and configured `app/app.py` (or environment variables) to use it.
-   The actual LLM loading code in `app/app.py` (using `ctransformers`) is uncommented and working.
-   You have API keys for different roles (as defined in `API_KEYS_ROLES` in `app/app.py`).
-   You have sample PDF, DOCX, and TXT files for testing ingestion and querying.
-   You have a tool for making API requests (e.g., `curl`, Postman, or custom scripts). The API docs are available at `http://localhost:8000/docs`.

## 2. Test Cases

### 2.1. API Key Authentication and Authorization

For each endpoint, attempt requests with:
    a.  No API Key. Expected: `401 Unauthorized` (or `403 Forbidden` if `auto_error=True` was used, but it's `False`).
    b.  An invalid/unknown API Key. Expected: `401 Unauthorized`.
    c.  A valid API Key that *does not* have the required role. Expected: `403 Forbidden`.
    d.  A valid API Key that *does* have the required role. Expected: Successful response (e.g., `200 OK`).

**Endpoints and Required Roles:**
*   **`/upload-document/`** (POST)
    *   Required Role: `ingestion_user` (or `admin`)
    *   Test with valid and invalid keys for this role.
*   **`/query-documents/`** (POST)
    *   Required Role: `query_user` (or `admin`)
    *   Test with valid and invalid keys for this role.
*   **`/summarize-text/`** (POST)
    *   Required Role: `query_user` (or `admin`)
    *   Test with valid and invalid keys for this role.
*   **`/generate-response/`** (POST)
    *   Required Role: `correspondence_user` (or `admin`)
    *   Test with valid and invalid keys for this role.

### 2.2. Document Ingestion (`/upload-document/`)

-   **Valid File Uploads:**
    -   Upload a sample PDF file. Expected: `200 OK` with success message. Verify in logs that chunks are added to ChromaDB.
    -   Upload a sample DOCX file. Expected: `200 OK`. Verify logs.
    -   Upload a sample TXT file. Expected: `200 OK`. Verify logs.
-   **Duplicate Uploads:**
    -   Re-upload the same file. Expected: `200 OK`. Verify logs indicate that chunks were likely skipped due to existing IDs. The number of items in ChromaDB should not significantly increase if chunking is deterministic.
-   **Unsupported File Type:**
    -   Attempt to upload an image file (e.g., JPG, PNG). Expected: Endpoint might return success (as it saves the file then tries to process), but logs in `document_processor.py` should indicate "Unsupported file type" and no chunks added. (Alternatively, the endpoint could be enhanced to check file types before saving).
-   **Empty File:**
    -   Upload an empty TXT file. Expected: Logs should indicate "No text extracted" or "No chunks created".
-   **Corrupted File (Optional, if you have samples):**
    -   Upload a corrupted PDF or DOCX. Expected: Graceful error handling in logs, API might return 500 or a specific error if identifiable.

### 2.3. Document Querying (`/query-documents/`)

*(Assumes documents have been successfully ingested)*
-   **Basic Query:**
    -   Send a query relevant to the content of your ingested documents. Expected: `200 OK`, LLM-generated answer, and list of retrieved context metadata.
    -   Verify that the answer seems relevant to the query and the retrieved context.
-   **Query with No Relevant Documents:**
    -   Send a query on a topic not covered in your documents. Expected: `200 OK`. The answer might be generic (e.g., "I couldn't find information about that..."), and the retrieved context might be empty or irrelevant.
-   **Vary `top_n`:**
    -   Test with different `top_n` values (e.g., 1, 3, 5) to see how the quantity of context affects the answer.
-   **Specific vs. Broad Queries:**
    -   Test with very specific queries (looking for a known detail) and broader queries (asking for overviews).
-   **Query after Multiple Document Uploads:**
    -   Upload several documents covering different topics. Query to see if the RAG system can pull context from the correct document(s).

### 2.4. Text Summarization (`/summarize-text/`)

-   **Short Text:**
    -   Provide a short paragraph. Expected: `200 OK`, a concise summary.
-   **Long Text:**
    -   Provide a long piece of text (e.g., several pages worth). Expected: `200 OK`, a summary.
    -   Observe if the `max_summary_length` hint is somewhat respected by the LLM.
-   **Highly Technical Text vs. General Text:**
    -   Test with different styles of text to see how the LLM performs.
-   **Text with No Clear Narrative (e.g., a list of items):**
    -   See how the LLM attempts to summarize it.

### 2.5. General Response Generation (`/generate-response/`)

-   **Drafting Correspondence:**
    -   Prompt: "Draft a short email to a colleague asking for an update on project X." Expected: `200 OK`, a reasonably drafted email.
-   **Consultation/Question Answering (without RAG from this endpoint):**
    -   Prompt: "What are the key principles of agile methodology?" Expected: `200 OK`, a general explanation from the LLM's own knowledge.
-   **Creative Generation:**
    -   Prompt: "Write a short poem about coding." Expected: `200 OK`, a poem.
-   **Vary Prompt Length and Complexity.**

## 3. Areas for Refinement

### 3.1. Prompt Engineering

-   **RAG Prompt (`/query-documents/`):**
    -   The current prompt is:
        ```
        Based on the following context, please answer the query.
        Context:
        {context_str}

        Query: {request.query}
        Answer:
        ```
    -   If answers are not satisfactory (e.g., LLM ignores context, or answers are too verbose/short), experiment with:
        -   Being more explicit: "Use *only* the provided context to answer the query. If the answer is not found in the context, state that."
        -   Adding instructions for answer length or style.
        -   Reordering (e.g., Query first, then Context).
-   **Summarization Prompt (`/summarize-text/`):**
    -   The current prompt is:
        ```
        Please summarize the following text concisely. Aim for a summary of around {request.max_summary_length} words if possible, but prioritize clarity and key information:

        Text to summarize:
        {request.text}

        Summary:
        ```
    -   If summaries are poor, try:
        -   Different phrasing: "Extract the key points and provide a brief summary...", "Summarize the following text in under X words:"
        -   Adding constraints: "Do not add new information."
-   **General LLM Parameters (Advanced):**
    -   When calling the LLM (e.g., `ctransformers`), models often support parameters like `temperature`, `top_k`, `top_p`, `max_new_tokens`. Adjusting these can significantly impact output quality (e.g., lower temperature for more factual, higher for more creative). This would require modifying the `llm(...)` call in `app/app.py`.

### 3.2. Text Chunking (`app/document_processor.py`)

-   The current manual chunker uses fixed `chunk_size` (1000) and `chunk_overlap` (200).
-   If RAG results are poor (e.g., context is too fragmented or too broad):
    -   Experiment with different `chunk_size` and `chunk_overlap` values.
    -   Consider more sophisticated chunking strategies (e.g., LangChain's `RecursiveCharacterTextSplitter` which is already in `requirements.txt`, or chunking by semantic similarity if performance allows).

### 3.3. Embedding Model

-   `all-MiniLM-L6-v2` is a good general-purpose model. If retrieval quality is an issue for very specific or technical domains, you might explore fine-tuning an embedding model or trying other pre-trained models, but this is a more advanced step.

### 3.4. Database Querying (Data Connectors)

-   Ensure connection strings and credentials for SQL Server/Oracle are handled securely if this moves beyond local testing (e.g., use environment variables, a secrets manager).
-   Test the `dataframe_to_text` conversion. For very large tables, sending the full table as text might be inefficient. Consider summarizing or selecting relevant parts of the table before converting to text for the LLM.

## 4. Reporting Issues

-   If you find bugs or areas where the assistant is not behaving as expected, note down:
    -   The endpoint used.
    -   The exact request payload (and any uploaded file names/types).
    -   The API key role used.
    -   The actual response received.
    -   The expected response or behavior.
    -   Any relevant logs from the Docker container (`docker logs <container_id>`).

This structured testing will help identify areas for improvement and ensure the AI assistant is robust and useful.
```
