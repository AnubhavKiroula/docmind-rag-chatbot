import requests
from typing import Dict, Any, Optional

# API BASE URL:
# Points to our running FastAPI backend on localhost.
BACKEND_URL = "http://localhost:8000"

# SEPARATION OF CONCERNS:
# We isolate network communication (HTTP requests to FastAPI) in this dedicated api_client.py module.
# This prevents our main app.py file from becoming cluttered with requests boilerplate, URL formatting,
# and HTTP error handling. If the API endpoints ever change, we only need to update this single file.

def send_query(
    query: str, 
    conversation_id: Optional[str] = None, 
    llm_mode: Optional[str] = None,
    top_k: Optional[int] = None
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
        "top_k": top_k
    }

    
    try:
        response = requests.post(url, json=payload, timeout=90.0)
        # Raise an exception for HTTP error status codes (e.g. 500, 404, etc.)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Connection Error: Unable to connect to the backend server. Please verify that the FastAPI backend is running on http://localhost:8000.",
            "confidence_score": 0.0,
            "sources": [],
            "conversation_id": conversation_id or ""
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
            "conversation_id": conversation_id or ""
        }
    except Exception as e:
        return {
            "answer": f"Unexpected Client Error: {str(e)}",
            "confidence_score": 0.0,
            "sources": [],
            "conversation_id": conversation_id or ""
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
            "error": "Unable to connect to the backend server. History could not be loaded."
        }
    except Exception as e:
        return {
            "conversation_id": conversation_id,
            "history": [],
            "error": f"Failed to retrieve conversation history: {str(e)}"
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
            "message": "Connection Error: Unable to connect to the backend server to upload PDF."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to ingest document: {str(e)}"
        }

