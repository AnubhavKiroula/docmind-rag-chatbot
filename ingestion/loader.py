import os
import pypdf
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

class CustomPDFReader:
    """
    A custom PDF document extractor using the pypdf library.
    Bypasses default LlamaIndex plain-text fallback parsing in environments 
    where llama-index-readers-file is not installed.
    """
    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        reader = pypdf.PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            metadata = extra_info.copy() if extra_info else {}
            metadata["page_number"] = i + 1
            docs.append(Document(text=text, metadata=metadata))
        return docs

def load_documents(folder_path: str) -> List[Document]:
    """
    Loads all PDF documents from the specified folder using CustomPDFReader.
    
    Parameters:
        folder_path (str): Path to the directory containing PDFs.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects containing the text contents of the PDFs.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The directory {folder_path} does not exist.")
        
    print(f"Loading documents from folder: {folder_path}...")
    
    # We configure SimpleDirectoryReader with our CustomPDFReader for PDF files.
    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".pdf"],
        file_extractor={".pdf": CustomPDFReader()}
    )
    
    documents = reader.load_data()
    return documents
