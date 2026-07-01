import uuid
import datetime
import os
import shutil
# We import Depends, UploadFile, File from FastAPI for database session and file uploading
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session

# Project root paths for file saving
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "sample")

# We import our request and response schemas from models.py for data validation.
from api.models import QueryRequest, QueryResponse, SourceChunk
# We import the pipeline components to drive retrieval and generation.
from query.retriever import QdrantRetriever
from query.llm_client import LLMClient
from query.synthesizer import ResponseSynthesizer

# We import database models and the session provider
from db.database import get_db
from db.models import Conversation, Message

# Create APIRouter instance to register endpoints.
router = APIRouter()

# Instantiate retriever, llm client, and synthesizer at the module level.
# This prevents recreating clients, downloading embedding weights, or connecting to databases on every HTTP request.
retriever = QdrantRetriever()
llm_client = LLMClient()
synthesizer = ResponseSynthesizer()

def get_utc_now_naive():
    """
    Helper to return a naive UTC datetime.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

@router.get("/health", summary="Health Check")
async def health_check():
    """
    GET endpoint that returns the API operational health status.
    GET is chosen because this is a read-only endpoint that does not modify system state.
    """
    return {"status": "ok", "timestamp": get_utc_now_naive().isoformat()}

@router.post("/query", response_model=QueryResponse, summary="Query PDF Index")
async def query_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    """
    POST endpoint that coordinates RAG logic: fetches semantically relevant chunks from Qdrant,
    passes them to the synthesizer LLM, updates conversation history, and returns the response.
    
    POST is chosen because:
    1. The payload contains query parameters and stateful context (conversation_id) in the request body.
    2. The endpoint alters the backend state by creating or updating chat logs.
    
    Persistence upgrade (Phase 3):
    - Replaced the temporary in-memory dictionary with SQLite + SQLAlchemy.
    - Conversation data now persists across server restarts and can be audited.
    """
    # (1) Resolve conversation ID: Use provided ID or generate a new UUID
    conversation_id = request.conversation_id
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        db_conversation = Conversation(id=conversation_id)
        db.add(db_conversation)
        db.commit()
    else:
        # Check if conversation exists, if not create it
        db_conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not db_conversation:
            db_conversation = Conversation(id=conversation_id)
            db.add(db_conversation)
            db.commit()
            
    # Retrieve previous conversation messages to construct history context
    previous_messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp).all()
    
    # (2) Fetch matching text chunks and relevance scores from Qdrant using the retriever
    requested_top_k = request.top_k
    try:
        retrieved_chunks = retriever.retrieve(request.query, top_k=requested_top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve contexts from Qdrant database: {str(e)}"
        )
        
    # (3) Augment the question with previous chat logs so the LLM retains conversation context
    history_context = ""
    if previous_messages:
        history_context = "Previous conversation context:\n"
        for msg in previous_messages:
            role_label = "User" if msg.role == "user" else "DocMind"
            history_context += f"{role_label}: {msg.content}\n"
        history_context += "\n--- End of Context ---\n"
        
    augmented_question = f"{history_context}Current Question: {request.query}"
    
    # (4) Synthesize the response and calculate confidence scores
    # If the client sent an explicit llm_mode override (e.g. from the UI toggle), 
    # we instantiate a custom client, otherwise reuse the default global client.
    active_llm_client = llm_client
    if request.llm_mode:
        active_llm_client = LLMClient(llm_mode=request.llm_mode)

    try:
        synthesis_result = synthesizer.synthesize(
            user_question=augmented_question,
            retrieved_chunks=retrieved_chunks,
            llm_client=active_llm_client
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM text generation failed: {str(e)}"
        )

    # (5) Save the user query and assistant response in SQLite database
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.query
    )
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=synthesis_result["answer"],
        confidence_score=synthesis_result["confidence_score"]
    )
    db.add(user_msg)
    db.add(assistant_msg)
    db.commit()
    
    # Map raw retrieved chunks list to our validated SourceChunk response schema
    sources = [
        SourceChunk(text=text, relevance_score=score) 
        for text, score in retrieved_chunks
    ]
    
    # Return validated QueryResponse payload
    return QueryResponse(
        answer=synthesis_result["answer"],
        confidence_score=synthesis_result["confidence_score"],
        sources=sources,
        conversation_id=conversation_id
    )

@router.get("/conversations/{conversation_id}", summary="Get Full Conversation History")
async def get_conversation_history(conversation_id: str, db: Session = Depends(get_db)):
    """
    GET endpoint to retrieve the full list of messages belonging to a conversation_id.
    Useful for reloading a past session in the Streamlit frontend.
    """
    db_conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not db_conversation:
        raise HTTPException(status_code=404, detail="Conversation session not found.")
        
    messages_list = [
        {
            "role": msg.role,
            "content": msg.content,
            "confidence_score": msg.confidence_score,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
        }
        for msg in db_conversation.messages
    ]
    
    return {
        "conversation_id": conversation_id,
        "history": messages_list
    }

@router.get("/history/{conversation_id}", summary="Get Chat History (Legacy)")
async def get_history(conversation_id: str, db: Session = Depends(get_db)):
    """
    GET endpoint to retrieve conversation logs (Legacy compatibility for older clients/tests).
    """
    return await get_conversation_history(conversation_id=conversation_id, db=db)

@router.post("/ingest", summary="Upload and Ingest a PDF Document")
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    POST endpoint that accepts a PDF file upload, saves it to the ingestion folder,
    and runs the parsing, chunking, and embedding ingestion pipeline programmatically.
    """
    # Verify file extension is PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF files are supported."
        )
        
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Save the file to the ingestion folder
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
        
    # Run the ingestion steps programmatically
    try:
        from ingestion.loader import load_documents
        from ingestion.chunker import chunk_documents
        from ingestion.embedder import get_embedding_model
        from ingestion.store import build_vector_index
        from config.settings import settings
        
        # 1. Load documents
        documents = load_documents(UPLOAD_DIR)
        if not documents:
            raise ValueError("No documents found after processing upload.")
            
        # 2. Chunk documents
        nodes = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        
        # 3. Load embedding model
        embed_model = get_embedding_model(settings.embed_model)
        
        # 4. Build vector index and store in Qdrant
        build_vector_index(
            nodes=nodes, 
            embed_model=embed_model, 
            qdrant_url=settings.qdrant_url, 
            collection_name=settings.qdrant_collection_name
        )
        
        # Reload the retriever's index dynamically so it knows about the new document chunks
        global retriever
        from query.retriever import QdrantRetriever
        retriever = QdrantRetriever()
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(nodes),
            "message": f"Successfully ingested '{file.filename}' and stored {len(nodes)} chunks in Qdrant."
        }
    except Exception as e:
        # Clean up the file if ingestion failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline failed: {str(e)}"
        )

