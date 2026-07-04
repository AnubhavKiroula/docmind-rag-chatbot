import os
import requests
from typing import Dict, Any, Optional

# DYNAMIC BACKEND URL RESOLUTION:
# WHY: Hardcoded 'http://localhost:8000' URLs break inside Docker containers and cloud hostings.
# Inside a Docker network or containerized environment, 'localhost' points to the container itself rather than companion services.
# Resolution order:
# 1. Streamlit Cloud secrets (st.secrets["BACKEND_URL"])
# 2. Environment variable (os.environ.get("BACKEND_URL")) for Docker networking (http://backend:8000)
# 3. Default fallback ("http://localhost:8000") for direct local development


def get_backend_url() -> str:
    """
    Dynamically resolves the active FastAPI backend endpoint URL.

    Returns:
        str: The resolved backend URL without a trailing slash.
    """
    # (1) Attempt to retrieve URL from Streamlit Cloud dashboard secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass

    # (2) Attempt to retrieve URL from environment variables (e.g., set in docker-compose)
    env_url = os.environ.get("BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    # (3) Fallback to local default URL
    return "http://localhost:8000"


BACKEND_URL = get_backend_url()

# SEPARATION OF CONCERNS:
# We isolate network communication (HTTP requests to FastAPI) in this dedicated api_client.py module.
# This prevents our main app.py file from becoming cluttered with requests boilerplate, URL formatting,
# and HTTP error handling. If the API endpoints ever change, we only need to update this single file.


def send_query(
    query: str,
    conversation_id: Optional[str] = None,
    llm_mode: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Submits a query to the FastAPI RAG engine and returns the response dictionary.

    Parameters:
        query (str): The question being asked.
        conversation_id (str): The active chat session ID (if continuing a thread).
        llm_mode (str): Active model choice override ('ollama' or 'groq').
        top_k (int): Optional dynamic retrieval limit (Top K).

    Returns:
        Dict[str, Any]: The JSON response containing 'answer', 'confidence_score', 'sources', etc.
    """
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "llm_mode": llm_mode,
        "top_k": top_k,
    }

    try:
        response = requests.post(url, json=payload, timeout=90.0)
        # Raise an exception for HTTP error status codes (e.g. 500, 404, etc.)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": f"Connection Error: Unable to connect to backend at {BACKEND_URL}. Please verify server status.",
            "confidence_score": 0.0,
            "sources": [],
            "conversation_id": conversation_id or "",
        }
    except requests.exceptions.HTTPError as http_err:
        try:
            err_msg = response.json().get("detail", str(http_err))
        except Exception:
            err_msg = str(http_err)
        return {
            "answer": f"API HTTP Error: {err_msg}",
            "confidence_score": 0.0,
            "sources": [],
            "conversation_id": conversation_id or "",
        }
    except Exception as e:
        return {
            "answer": f"Unexpected Client Error: {str(e)}",
            "confidence_score": 0.0,
            "sources": [],
            "conversation_id": conversation_id or "",
        }


def get_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """
    Retrieves the list of past messages for a given session ID.

    Parameters:
        conversation_id (str): The session UUID.

    Returns:
        Dict[str, Any]: A dictionary containing conversation_id and a list of message logs.
    """
    url = f"{BACKEND_URL}/conversations/{conversation_id}"

    try:
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "conversation_id": conversation_id,
            "history": [],
            "error": f"Unable to connect to backend at {BACKEND_URL}. History could not be loaded.",
        }
    except Exception as e:
        return {
            "conversation_id": conversation_id,
            "history": [],
            "error": f"Failed to retrieve conversation history: {str(e)}",
        }


def upload_pdf(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    POSTs a raw PDF file as multipart/form-data to the FastAPI backend /ingest endpoint.

    Parameters:
        file_bytes (bytes): The raw bytes of the uploaded PDF file.
        filename (str): The name of the file.

    Returns:
        Dict[str, Any]: JSON response status, chunk count, and message.
    """
    url = f"{BACKEND_URL}/ingest"
    files = {"file": (filename, file_bytes, "application/pdf")}

    try:
        response = requests.post(url, files=files, timeout=120.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": f"Connection Error: Unable to connect to backend at {BACKEND_URL} to upload PDF.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to ingest document: {str(e)}"}
