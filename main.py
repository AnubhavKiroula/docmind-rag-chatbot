# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ 
# based on standard Python type hints. It is asynchronous out-of-the-box and generates interactive 
# Swagger UI documentation automatically.
#
# CORS (Cross-Origin Resource Sharing) middleware is required to allow frontend web applications 
# running on different origins (like a React app on http://localhost:3000) to safely make HTTP requests 
# to our FastAPI backend on http://localhost:8000. Without this, modern web browsers will block the API requests.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# We import our registered router containing health check, history, and query endpoints.
from api.routes import router as api_router

# Initialize the FastAPI app instance with metadata for interactive Swagger documentation.
app = FastAPI(
    title="DocMind RAG Chatbot",
    description="An intelligent retrieval-augmented generation (RAG) backend to chat with your ingested PDF documents.",
    version="1.0.0"
)

# Define allowed origins for CORS.
# Allows both the standard localhost React dev port (3000) and the backend self-port (8000).
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

# Register the CORSMiddleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers (Content-Type, Authorization, etc.)
)

# Include the endpoints router. All endpoint calls will be routed through this.
app.include_router(api_router)

@app.get("/", summary="Root Endpoint")
async def root():
    """
    GET request on the root URL. Returns a welcome message with instructions on how to query.
    """
    return {
        "message": "Welcome to DocMind. POST to /query with {'query': 'your question'} or GET /docs for Swagger UI documentation."
    }
