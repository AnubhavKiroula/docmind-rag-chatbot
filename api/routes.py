import uuid
import datetime
# We import APIRouter and HTTPException from FastAPI to build our routing system.
from fastapi import APIRouter, HTTPException
# We import our request and response schemas from models.py for data validation.
from api.models import QueryRequest, QueryResponse, SourceChunk
# We import the pipeline components to drive retrieval and generation.
from query.retriever import QdrantRetriever
from query.llm_client import LLMClient
from query.synthesizer import ResponseSynthesizer

# Create APIRouter instance to register endpoints.
router = APIRouter()

# CONVERSATION HISTORY:
# Keeping track of previous user queries and assistant responses is key to enabling follow-up questions
# (e.g. asking "Who is the author?" followed by "What else did they write?"). 
# Without history, the system is stateless and forgets the context immediately.
#
# IN-MEMORY STORAGE:
# We store history in an in-memory dictionary for Phase 2 because it is fast to implement, has no setup overhead,
# and is sufficient for development. In Phase 3, we will migrate this to a persistent database (like PostgreSQL
# or Redis) so history survives backend server restarts.
conversation_store = {}

# Instantiate retriever, llm client, and synthesizer at the module level.
# This prevents recreating clients, downloading embedding weights, or connecting to databases on every HTTP request.
retriever = QdrantRetriever()
llm_client = LLMClient()
synthesizer = ResponseSynthesizer()

@router.get("/health", summary="Health Check")
async def health_check():
    """
    GET endpoint that returns the API operational health status.
    GET is chosen because this is a read-only endpoint that does not modify system state.
    """
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

@router.post("/query", response_model=QueryResponse, summary="Query PDF Index")
async def query_endpoint(request: QueryRequest):
    """
    POST endpoint that coordinates RAG logic: fetches semantically relevant chunks from Qdrant,
    passes them to the synthesizer LLM, updates conversation history, and returns the response.
    
    POST is chosen because:
    1. The payload contains query parameters and stateful context (conversation_id) in the request body.
    2. The endpoint alters the backend state by creating or updating in-memory chat logs.
    """
    # (1) Resolve conversation ID: Use provided ID or generate a new UUID
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Retrieve previous conversation exchanges to construct history context
    previous_exchanges = conversation_store.get(conversation_id, [])
    
    # (2) Fetch matching text chunks and relevance scores from Qdrant using the retriever
    try:
        retrieved_chunks = retriever.retrieve(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve contexts from Qdrant database: {str(e)}"
        )
        
    # (3) Augment the question with previous chat logs so the LLM retains conversation context
    history_context = ""
    if previous_exchanges:
        history_context = "Previous conversation context:\n"
        for exchange in previous_exchanges:
            history_context += f"User: {exchange['query']}\nDocMind: {exchange['answer']}\n"
        history_context += "\n--- End of Context ---\n"
        
    augmented_question = f"{history_context}Current Question: {request.query}"
    
    # (4) Synthesize the response and calculate confidence scores
    try:
        synthesis_result = synthesizer.synthesize(
            user_question=augmented_question,
            retrieved_chunks=retrieved_chunks,
            llm_client=llm_client
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM text generation failed: {str(e)}"
        )

    # (5) Save the exchange in our in-memory history log
    timestamp = datetime.datetime.utcnow().isoformat()
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []
        
    conversation_store[conversation_id].append({
        "query": request.query,
        "answer": synthesis_result["answer"],
        "timestamp": timestamp
    })
    
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

@router.get("/history/{conversation_id}", summary="Get Chat History")
async def get_history(conversation_id: str):
    """
    GET endpoint to retrieve the conversation history log for a specific session ID.
    """
    if conversation_id not in conversation_store:
        raise HTTPException(status_code=404, detail="Conversation session not found.")
    return {
        "conversation_id": conversation_id,
        "history": conversation_store[conversation_id]
    }
