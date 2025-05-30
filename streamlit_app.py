import streamlit as st
import requests
import os

# --- Configuration ---
FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
API_DOCS_URL = f"{FASTAPI_BASE_URL}/docs"

# --- Page Setup ---
st.set_page_config(page_title="AI Assistant", layout="wide", page_icon="🤖")

st.title("🤖 Local AI Assistant")
st.caption(f"Powered by a FastAPI backend. API Docs available at [{API_DOCS_URL}]({API_DOCS_URL})")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today? Please enter your API key and select a mode to get started."}]
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Chat with Documents (RAG)" # Default mode
if "api_key_confirmed" not in st.session_state:
    st.session_state.api_key_confirmed = False # To avoid asking for API key repeatedly if already entered

# --- Helper Function to Call FastAPI Backend ---
def call_fastapi_backend(endpoint: str, payload: dict, api_key: str) -> dict:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", json=payload, headers=headers, timeout=60)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        try:
            # Try to parse error from FastAPI
            error_detail = response.json().get("detail", str(http_err))
        except:
            error_detail = str(http_err)
        return {"error": f"HTTP error: {error_detail}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Request error: {req_err}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

# --- UI Components ---

# API Key Input (conditionally displayed or in sidebar)
with st.expander("API Key Configuration", expanded=not st.session_state.api_key_confirmed):
    new_api_key = st.text_input("Enter your API Key:", type="password", value=st.session_state.api_key)
    if st.button("Save API Key"):
        if new_api_key:
            st.session_state.api_key = new_api_key
            st.session_state.api_key_confirmed = True # Mark as confirmed
            st.success("API Key saved! You can now close this section.")
            st.rerun() # Rerun to reflect changes, e.g., close expander
        else:
            st.warning("Please enter an API Key.")

# Mode Selector
st.session_state.selected_mode = st.radio(
    "Choose Interaction Mode:",
    ("Chat with Documents (RAG)", "General Chat / Consultation"),
    index=0 if st.session_state.selected_mode == "Chat with Documents (RAG)" else 1,
    key="mode_selector_radio" # Added key for stability
)
st.markdown("---")


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Your message:"):
    if not st.session_state.api_key:
        st.warning("Please enter and save your API Key in the configuration section above before sending a message.")
        # Optionally, add this message to chat history as well
        # st.session_state.messages.append({"role": "assistant", "content": "Please enter and save your API Key first!"})
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call backend based on selected mode
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = None
                if st.session_state.selected_mode == "Chat with Documents (RAG)":
                    payload = {"query": prompt, "top_n": 3} # top_n can be made configurable later
                    response_data = call_fastapi_backend("/query-documents/", payload, st.session_state.api_key)
                elif st.session_state.selected_mode == "General Chat / Consultation":
                    payload = {"prompt": prompt}
                    response_data = call_fastapi_backend("/generate-response/", payload, st.session_state.api_key)

                if response_data:
                    if "error" in response_data:
                        assistant_response = f"Error: {response_data['error']}"
                        st.error(assistant_response)
                    elif st.session_state.selected_mode == "Chat with Documents (RAG)":
                        assistant_response = response_data.get("answer", "Sorry, I couldn't get an answer.")
                        # Optionally display context sources if desired
                        # sources = response_data.get("retrieved_context_metadatas", [])
                        # if sources:
                        #    assistant_response += "\n\n*Sources:*\n"
                        #    for i, source_meta in enumerate(sources):
                        #        assistant_response += f"- {source_meta.get('source', 'Unknown')} (Chunk {source_meta.get('chunk_index', 'N/A')})\n"
                    else: # General Chat
                        assistant_response = response_data.get("generated_response", "Sorry, I couldn't generate a response.")

                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    st.error("No response received from the backend.")
                    st.session_state.messages.append({"role": "assistant", "content": "No response received from backend."})
