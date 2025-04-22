from typing import List, Dict, Any, Optional
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

class EmbeddingGenerator:
    """Utility class to generate embeddings for documents."""
    
    @staticmethod
    def get_bge_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
        """
        Get Hugging Face BGE embeddings model.
        
        Args:
            model_name: Name of the Hugging Face model.
            
        Returns:
            Hugging Face embeddings model.
        """
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    @staticmethod
    def get_embeddings_model(model_type: str = "bge", model_name: Optional[str] = None):
        """
        Get embeddings model based on type.
        
        Args:
            model_type: Type of embeddings model (bge or huggingface).
            model_name: Name of the specific model.
            
        Returns:
            Embeddings model.
        """
        if model_type.lower() in ["huggingface", "bge"]:
            return EmbeddingGenerator.get_bge_embeddings(
                model_name or "BAAI/bge-small-en-v1.5"
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}") 