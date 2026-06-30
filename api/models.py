from pydantic import BaseModel, Field
from typing import List, Optional

# In FastAPI, Pydantic models are used for three main purposes:
# 1. Request Validation: Ensuring incoming JSON payloads contain required fields and correct data types.
# 2. Response Serialization: Filtering and formatting outgoing JSON data.
# 3. Auto-Documentation: Generating interactive API documentation (Swagger UI) at `/docs` out of the box.

class QueryRequest(BaseModel):
    """
    Schema for incoming client query requests.
    """
    query: str = Field(
        ..., 
        description="The search question or query text to submit to the RAG system.",
        examples=["What are the system specifications for running this project?"]
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier to associate the query with an existing chat history context.",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )

class SourceChunk(BaseModel):
    """
    Schema representing a single retrieved document chunk used to construct the answer.
    """
    text: str = Field(
        ...,
        description="The raw text snippet extracted from the PDF."
    )
    relevance_score: float = Field(
        ...,
        description="The semantic similarity score of this chunk relative to the user query (closer to 1 is better)."
    )

class QueryResponse(BaseModel):
    """
    Schema for structured JSON responses sent back to the client.
    """
    answer: str = Field(
        ...,
        description="The final natural language response synthesized by the LLM citing document sources."
    )
    confidence_score: float = Field(
        ...,
        description="The calculated average relevance score of the retrieved sources (scaled 0-1)."
    )
    sources: List[SourceChunk] = Field(
        ...,
        description="List of document text chunks used by the synthesizer to construct the response."
    )
    conversation_id: str = Field(
        ...,
        description="The conversation identifier. If not provided in the request, a new one is generated and returned."
    )
