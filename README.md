# DocMind — Intelligent Document Assistant

DocMind is a Retrieval-Augmented Generation (RAG) chatbot designed to ingest PDF documents, perform offline embedding extraction, store them in a local vector database, and provide intelligent question-answering based on the document context.

---

## Tech Stack
- **LlamaIndex**: Main orchestration framework used for loading, chunking, and indexing.
- **Qdrant**: High-performance open-source vector database used to store and perform similarity search on embeddings.
- **sentence-transformers/all-MiniLM-L6-v2**: A lightweight (80MB) pre-trained embedding model that runs fully offline.
- **Docker**: Used to run Qdrant locally in a container with zero manual installation.
- **Python 3.10+ & virtualenv**: Main language and package isolation tool.
- **python-dotenv**: Configuration manager that reads from the local `.env` file.
- **pypdf**: PDF reading engine integrated with LlamaIndex.
- **pytest**: Core testing framework to validate the pipeline stages.

---

## Phase 1: Ingestion Pipeline
In Phase 1, we built the core scaffold and ingestion pipeline. It automates processing raw PDF files from the local filesystem into queryable vector representation inside Qdrant.

### Ingestion Flow Diagram
```text
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
```

---

## Local Setup

### Prerequisites
- Docker Desktop installed and running.
- Python 3.10+ installed.

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

6. **Place PDFs and Run Ingestion**
   Place any PDF document you want to ingest inside the `data/sample/` folder (a default `sample.pdf` is provided for testing).
   Run the ingestion script:
   ```bash
   python scripts/ingest.py
   ```

7. **Verify Collections in Qdrant**
   Open [http://localhost:6333/dashboard](http://localhost:6333/dashboard), click on **Collections**, and select `docmind_docs` to verify that your points and vectors are populated.

---

## Running Tests
Run the automated pytest test suite to verify code compliance and correctness:
```bash
python -m pytest
```
