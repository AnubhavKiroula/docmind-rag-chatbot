from typing import List
# We import SentenceSplitter from LlamaIndex core's node_parser module.
# SentenceSplitter breaks text down into smaller sections without breaking sentences in half, preserving logical meaning.
from llama_index.core.node_parser import SentenceSplitter
# We import Document and BaseNode from LlamaIndex core's schema.
# Chunks are represented as BaseNode objects in LlamaIndex.
from llama_index.core.schema import Document, BaseNode

def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[BaseNode]:
    """
    Splits document text into smaller, overlapping chunks (nodes) using SentenceSplitter.
    
    Parameters:
        documents (List[Document]): List of input documents to split.
        chunk_size (int): The target size of each text chunk in tokens.
        chunk_overlap (int): The number of overlapping tokens between adjacent chunks.
        
    Returns:
        List[BaseNode]: A list of text nodes representing the chunks.
    """
    # Print status indicating that chunking is starting.
    print(f"Splitting {len(documents)} document(s) into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    
    # We instantiate SentenceSplitter with the specified chunk size and overlap.
    # Chunk overlap ensures the end of one chunk is repeated at the start of the next chunk.
    # This overlap prevents semantic fragmentation and context loss near the boundaries.
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # We pass our list of documents to get_nodes_from_documents.
    # This parses all documents, splits them, and extracts standard TextNodes representing chunks.
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Return the list of chunked nodes.
    return nodes
