# DocMind — Intelligent Document Assistant

DocMind is a Retrieval-Augmented Generation (RAG) chatbot designed to ingest PDF documents, perform offline embedding extraction, store them in a local vector database, and provide intelligent question-answering based on the document context.

---

## Tech Stack
- **LlamaIndex**: Main orchestration framework used for loading, chunking, and indexing.
- **Qdrant**: High-performance open-source vector database used to store and perform similarity search on embeddings.
- **sentence-transformers/all-MiniLM-L6-v2**: A lightweight (80MB) pre-trained embedding model that runs fully offline.
- **FastAPI**: Lightweight, asynchronous web framework for building standard REST APIs with auto-generated documentation.
- **Ollama**: Local LLM inference engine running models locally (e.g. `llama3.2`).
- **Groq API**: High-speed cloud fallback for LLM completions.
- **Docker**: Used to run Qdrant locally in a container with zero manual installation.
- **Python 3.12+ & virtualenv**: Main language and package isolation tool.
- **pytest**: Core testing framework to validate the pipeline stages.

---

## System Architecture

Below is the complete architectural flow showing document ingestion, vector storage, and query processing.

```text
                      [ INGESTION PIPELINE ]
                      
  +------------------+      +---------------------+      +----------------------+
  |  data/sample/    | ---> |   ingestion/loader  | ---> |  ingestion/chunker   |
  |  (PDF files)     |      | SimpleDirectoryReader|      |  (SentenceSplitter)  |
  +------------------+      +---------------------+      +----------------------+
                                                                    |
                                                                    v
  +------------------+      +---------------------+      +----------------------+
  |  Qdrant DB       | <--- |   ingestion/store   | <--- |  ingestion/embedder  |
  | (Vector Storage) |      | (VectorStoreIndex)  |      | (all-MiniLM-L6-v2)   |
  +------------------+      +---------------------+      +----------------------+
  
  
                      [ QUERY & INFERENCE PIPELINE ]
                      
  +------------------+      +---------------------+
  |   User Client    | ---> |    FastAPI Routes   |
  | (HTTP POST Request)     |   (/query endpoint)  |
  +------------------+      +---------------------+
                                       |
                                       v
                            +---------------------+      +----------------------+
                            |   QdrantRetriever   | ---> |  Qdrant Database     |
                            | (Retrieve top_k=3)  | <--- | (Similarity Search)  |
                            +---------------------+      +----------------------+
                                       |
                                       v (chunks + relevance)
                            +---------------------+
                            | ResponseSynthesizer |
                            | (Prompt & Confidence|
                            |  Score Computation) |
                            +---------------------+
                                       |
                                       v (grounded prompt)
                            +---------------------+      +----------------------+
                            |      LLMClient      | ---> |  Ollama (llama3.2)   |
                            | (Routing/Fallback)  |      |  (Local inference)   |
                            +---------------------+      +----------------------+
                                       |                            | (if offline)
                                       v (answer text)              v
                            +---------------------+      +----------------------+
                            |     User Client     | <--- |  Groq Cloud API      |
                            | (JSON Response API) |      |  (Cloud fallback)    |
                            +---------------------+      +----------------------+
```

---

## Phase 1: Ingestion Pipeline
In Phase 1, we built the core ingestion pipeline. It automates processing raw PDF files from the local filesystem into queryable vector representation inside Qdrant.

For detailed documentation on Phase 1 ingestion, see the [Ingestion Documentation](file:///d:/PROJECT/docmind-rag-chatbot/ingestion/__init__.py).

---

## Phase 2: Query Engine & FastAPI Backend
In Phase 2, we built a retrieval-augmented query engine and exposed it via a REST API backend. Key accomplishments include:
1. **Semantic Retriever**: Connections to Qdrant to perform vector similarity queries.
2. **Unified LLM Interface**: Seamless routing between a local Ollama model (`llama3.2`) and Groq cloud inference with automatic fallback if local Ollama is offline.
3. **Response Synthesizer**: Prompt assembly, citation generator, and confidence score scoring (derived from average chunk similarity).
4. **FastAPI Endpoints**: Operates `/health`, `/query`, and `/history` endpoints with JSON payload validation.
5. **Session History**: In-memory context tracking via `conversation_id` so that the LLM remembers previous dialogue turns.

---

## Local Setup

### Prerequisites
- Docker Desktop installed and running.
- Python 3.12+ installed.
- Ollama running locally.
- Groq Cloud API Key (Optional fallback, from [console.groq.com](https://console.groq.com)).

> [!IMPORTANT]
> **Ollama Model Verification**: Before starting the backend, verify that you have downloaded the target local model by running:
> ```bash
> ollama list
> ```
> Confirm that `llama3.2` (or the model name configured in your `.env` as `LOCAL_LLM_MODEL`) is present in the list. This prevents runtime errors and avoids fallback issues if you have multiple local models installed.

### Installation & Run Steps

1. **Clone and Navigate to Project**
   ```bash
   cd docmind-rag-chatbot
   ```

2. **Initialize and Activate Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Qdrant Vector DB**
   Ensure Docker Desktop is running, then start the container:
   ```bash
   docker compose up -d
   ```
   Verify that Qdrant is running by accessing the web dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

5. **Configure Environment Variables**
   Copy the sample environment file to `.env`:
   ```bash
   cp .env.example .env
   ```
   Fill in your `GROQ_API_KEY` (if using Groq API) and set `LLM_MODE` to `ollama` (for local) or `groq` (for cloud API).

6. **Place PDFs and Run Ingestion**
   Place any PDF document you want to ingest inside the `data/sample/` folder.
   Run the ingestion script:
   ```bash
   python scripts/ingest.py
   ```

7. **Start FastAPI Backend Server**
   Start the development server with live reload:
   ```bash
   python -m uvicorn main:app --reload
   ```
   The API will be available at [http://localhost:8000](http://localhost:8000).
   The interactive API docs will be at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Returns 200 OK with server timestamp to verify operations.
- **Example curl**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Example Response**:
  ```json
  {
    "status": "ok",
    "timestamp": "2026-06-30T11:00:00.000000"
  }
  ```

### 2. Query PDF Index (RAG)
- **Endpoint**: `POST /query`
- **Description**: Submits a user query, fetches relevant chunks from Qdrant, synthesizes a cited response, logs to session history, and returns JSON.
- **Example curl**:
  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"What is the main topic of the documents?\"}"
  ```
- **Example Response**:
  ```json
  {
    "answer": "Based on [Chunk 1], the main topic is the implementation of an ingestion pipeline. [Chunk 2] specifies that Qdrant is used as the vector store.",
    "confidence_score": 0.8542,
    "sources": [
      {
        "text": "The ingestion pipeline extracts pages from the PDF...",
        "relevance_score": 0.8923
      },
      {
        "text": "We store these points into a collection named docmind_docs in Qdrant...",
        "relevance_score": 0.8161
      }
    ],
    "conversation_id": "31e845c4-7221-4f13-bb17-802f067d26da"
  }
  ```

### 3. Get Conversation History
- **Endpoint**: `GET /history/{conversation_id}`
- **Description**: Retrieves history logs for a specific session.
- **Example curl**:
  ```bash
  curl http://localhost:8000/history/31e845c4-7221-4f13-bb17-802f067d26da
  ```

---

## Conversation Flow & History
1. If no `conversation_id` is passed in `POST /query`, the backend generates a new UUID and returns it.
2. In subsequent queries, send the `conversation_id` in the request payload:
   ```json
   {
     "query": "Can you elaborate on that?",
     "conversation_id": "31e845c4-7221-4f13-bb17-802f067d26da"
   }
   ```
3. The API retrieves previous dialogue turns for that ID, appends them to the context, and queries the LLM.

---

## LLM Configuration (Local vs. Cloud)
You can customize model parameters in your `.env` file:
```ini
LLM_MODE=ollama          # 'ollama' for local, 'groq' for cloud API
LOCAL_LLM_MODEL=llama3.2 # Local Ollama model name
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=gsk_...     # Optional cloud API key
```
- If `LLM_MODE=ollama` and Ollama is offline, the client will auto-fallback to Groq if a `GROQ_API_KEY` is present.

---

## Scaling & Production Considerations (Phase 3 & 4 Planning)
To prepare the DocMind RAG Chatbot for production and scaling, the following middleware upgrades are planned:
- **FastAPI Request Logging Middleware**: Standardized JSON logs for all incoming requests, response times, and processing durations using libraries like `structlog`.
- **FastAPI Rate Limiting Middleware**: IP-based rate limiting (using Redis and `slowapi`) on the `POST /query` endpoint to prevent brute-forcing and resource starvation.
- **Persistent Session Stores**: Swapping the in-memory chat dict with a PostgreSQL or Redis backend to persist conversation histories.

---

## Running Tests
Run the automated pytest test suite to verify code compliance and correctness. **Note**: Make sure to activate the virtual environment (`venv`) before running tests so that packages and the local python path are correctly set up.
```bash
# Activate environment first:
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

python -m pytest
```
