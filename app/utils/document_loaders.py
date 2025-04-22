import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document

class DocumentLoader:
    """Utility class to load documents from different sources."""
    
    @staticmethod
    def load_from_directory(directory_path: str) -> List[Document]:
        """
        Load documents from a directory containing different file types.
        
        Args:
            directory_path: Path to the directory containing documents.
            
        Returns:
            List of Document objects.
        """
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.glob("**/*"):
            if file_path.is_file():
                try:
                    file_extension = file_path.suffix.lower()
                    
                    if file_extension == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        documents.extend(loader.load())
                    
                    elif file_extension == ".md":
                        loader = UnstructuredMarkdownLoader(str(file_path))
                        documents.extend(loader.load())
                    
                    elif file_extension in [".txt", ".log", ".csv"]:
                        loader = TextLoader(str(file_path))
                        documents.extend(loader.load())
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    @staticmethod
    def load_from_files(file_paths: List[str]) -> List[Document]:
        """
        Load documents from a list of file paths.
        
        Args:
            file_paths: List of paths to files.
            
        Returns:
            List of Document objects.
        """
        documents = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            try:
                file_extension = path.suffix.lower()
                
                if file_extension == ".pdf":
                    loader = PyPDFLoader(str(path))
                    documents.extend(loader.load())
                
                elif file_extension == ".md":
                    loader = UnstructuredMarkdownLoader(str(path))
                    documents.extend(loader.load())
                
                elif file_extension in [".txt", ".log", ".csv"]:
                    loader = TextLoader(str(path))
                    documents.extend(loader.load())
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents 