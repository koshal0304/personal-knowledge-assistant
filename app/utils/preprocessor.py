from typing import List, Dict, Any
import re

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentPreprocessor:
    """
    Utility class for document preprocessing tasks like cleaning and chunking.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the text by removing extra whitespace, fixing line breaks, etc.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Fix common character encoding issues
        text = text.replace('Â', '').replace('â', '').replace('â€™', "'")
        
        return text.strip()
    
    @staticmethod
    def chunk_documents(
        documents: List[Document], 
        chunk_size: int = 512, 
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Split documents into chunks of specified size.
        
        Args:
            documents: List of documents to chunk.
            chunk_size: Size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
            
        Returns:
            List of chunked documents.
        """
        # Create a text splitter with the specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Clean the document texts
        for i, doc in enumerate(documents):
            if doc.page_content:
                documents[i].page_content = DocumentPreprocessor.clean_text(doc.page_content)
        
        # Split the documents into chunks
        chunked_documents = text_splitter.split_documents(documents)
        
        return chunked_documents 