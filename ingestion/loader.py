import os
from typing import List
# We import SimpleDirectoryReader from LlamaIndex core.
# This utility is designed to load files from a specific directory automatically.
from llama_index.core import SimpleDirectoryReader
# We import Document, which is LlamaIndex's standard data structure representing text documents.
from llama_index.core.schema import Document

def load_documents(folder_path: str) -> List[Document]:
    """
    Loads all PDF documents from the specified folder.
    
    Parameters:
        folder_path (str): Path to the directory containing PDFs.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects containing the text contents of the PDFs.
    """
    # Check if the directory exists, and raise an error if it doesn't.
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The directory {folder_path} does not exist.")
        
    # We print a message indicating which folder we are loading PDFs from.
    print(f"Loading documents from folder: {folder_path}...")
    
    # SimpleDirectoryReader scans the folder and reads every PDF it finds.
    # We configure it with required_exts=[".pdf"] to ensure it only processes PDF files.
    reader = SimpleDirectoryReader(input_dir=folder_path, required_exts=[".pdf"])
    
    # We call load_data() which triggers the reader to parse all PDF files.
    # Each PDF page or document will be loaded into a Document object.
    documents = reader.load_data()
    
    # Return the loaded documents.
    return documents
