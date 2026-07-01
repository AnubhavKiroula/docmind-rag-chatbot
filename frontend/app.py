import streamlit as st
from api_client import send_query, get_conversation_history, upload_pdf

# PAGE CONFIGURATION
st.set_page_config(
    page_title="DocMind — Chat with your Documents",
    page_icon="🤖",
    layout="wide"
)

# Title & Description
st.title("🤖 DocMind — Chat with your Documents")
st.markdown("An intelligent document assistant powered by **LlamaIndex**, **Qdrant**, and **Ollama/Groq**.")

# STREAMLIT STATE PERSISTENCE:
# Streamlit runs the entire Python file from top to bottom on every user interaction (clicks, typing, etc.).
# To prevent losing data (such as the current conversation ID or message history), we store them in `st.session_state`.
# `st.session_state` is a dictionary-like object that persists across script reruns for the user's browser session.

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR OPTIONS
st.sidebar.title("Configuration")

# LLM Mode Selector (Step 8)
# Giving users the choice to toggle between local models and cloud API builds user trust 
# and makes it easy to compare speed/quality tradeoffs in real-time.
llm_option = st.sidebar.selectbox(
    "Active LLM Mode",
    options=["Ollama (Local)", "Groq (API)"],
    index=0
)

# Convert UI option to config keys ('ollama' or 'groq')
llm_mode = "ollama" if llm_option == "Ollama (Local)" else "groq"

# st.session_state.llm_mode is used to pass through to send_query()
st.session_state.llm_mode = llm_mode

# Retrieval Limit (Top K) Slider (Phase 4)
# Allows the user to dynamically adjust the number of document chunks fetched from Qdrant,
# which directly controls the retrieval context size.
top_k = st.sidebar.slider(
    "Retrieval Limit (Top K)",
    min_value=1,
    max_value=15,
    value=3,
    step=1,
    help="Adjust the number of document chunks retrieved from the database to answer the query. Increase this to answer global questions about the entire document."
)
st.session_state.top_k = top_k


# Start New Conversation Button (Step 9)
# Resetting the context is critical so that the user can switch topics or clear history
# without restarting the FastAPI server or closing their browser window.
if st.sidebar.button("➕ Start New Conversation", use_container_width=True):
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.success("Started a new conversation session!")
    st.rerun()

# Document Ingestion Instruction Reminder (Step 9)
# This provides clear guidance for newcomers or developers to understand how to feed
# files into the RAG system before querying, preventing 'no data found' confusion.
st.sidebar.info(
    "💡 **Add Documents (CLI)**:\n"
    "1. Drop your `.pdf` documents into `data/sample/`.\n"
    "2. Run the ingestion pipeline:\n"
    "   `python scripts/ingest.py`"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload PDF (Web)")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF file to index it",
    type=["pdf"],
    help="Upload a new document to the RAG database to query its content immediately."
)

if uploaded_file is not None:
    if st.sidebar.button("⚙️ Ingest Document", use_container_width=True):
        with st.sidebar.status("Processing PDF...") as status:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            
            # Send file bytes and name to backend
            res = upload_pdf(file_bytes, filename)
            
            if res.get("status") == "success":
                status.update(label=f"Ingested: {res['filename']} ({res['chunks_created']} chunks)", state="complete")
                st.sidebar.success(res["message"])
            else:
                status.update(label="Ingestion failed", state="error")
                st.sidebar.error(res.get("message", "Unknown error occurred."))


# RENDER CHAT HISTORY
# Display all existing messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display metadata (confidence & sources) only for assistant responses (Step 7)
        # Showing confidence scores builds trust by indicating how well-grounded the response is,
        # and listing the sources explicitly ensures transparency.
        if message["role"] == "assistant":
            if "confidence_score" in message and message["confidence_score"] is not None:
                st.caption(f"**Confidence**: {int(message['confidence_score'] * 100)}%")
                
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources used"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(f"**Source {idx + 1}**:")
                        st.write(src)

# CHAT INPUT BOX
if prompt := st.chat_input("Ask a question about your ingested documents..."):
    # (1) Display the user message instantly
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # (2) Query backend API through our client
    with st.spinner("Thinking..."):
        response = send_query(
            query=prompt, 
            conversation_id=st.session_state.conversation_id,
            llm_mode=st.session_state.llm_mode,
            top_k=st.session_state.top_k
        )

        
    # (3) Update the session ID (if newly generated by the backend)
    if "conversation_id" in response and response["conversation_id"]:
        st.session_state.conversation_id = response["conversation_id"]
        
    # (4) Display the assistant response
    with st.chat_message("assistant"):
        st.write(response["answer"])
        
        confidence = response.get("confidence_score")
        if confidence is not None:
            st.caption(f"**Confidence**: {int(confidence * 100)}%")
            
        sources_list = response.get("sources", [])
        # If sources are returned, format them as clean strings
        # (send_query parses response.sources into JSON list, we extract text fields)
        formatted_sources = []
        if sources_list:
            with st.expander("📚 Sources used"):
                for idx, src in enumerate(sources_list):
                    st.markdown(f"**Source {idx + 1}**:")
                    # If sources are dictionaries (SourceChunk model) or raw strings
                    text_content = src["text"] if isinstance(src, dict) else src
                    st.write(text_content)
                    formatted_sources.append(text_content)
                    
    # Save assistant message to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "confidence_score": confidence,
        "sources": formatted_sources
    })
