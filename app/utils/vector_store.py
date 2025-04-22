from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import time

from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec, Metric

from .embeddings import EmbeddingGenerator

# Load environment variables
load_dotenv()

class VectorStore:
    """Utility class to manage vector store operations."""
    
    @staticmethod
    def get_pinecone_client():
        """Initialize and return Pinecone client."""
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("Pinecone API key must be set.")
        
        return PineconeClient(api_key=api_key)
    
    @staticmethod
    def delete_index(index_name: str):
        """
        Delete a Pinecone index if it exists.
        
        Args:
            index_name: Name of the index to delete.
        """
        # Initialize Pinecone
        pinecone_client = VectorStore.get_pinecone_client()
        
        # List all indexes
        indexes = pinecone_client.list_indexes()
        
        # Check if the index exists
        if any(index.name == index_name for index in indexes):
            # Delete the index
            pinecone_client.delete_index(index_name)
            print(f"Deleted Pinecone index: {index_name}")
            # Sleep to ensure the deletion completes
            time.sleep(5)
    
    @staticmethod
    def get_or_create_index(index_name: str, dimension: int = 384, force_recreate: bool = False):
        """
        Get existing Pinecone index or create a new one.
        
        Args:
            index_name: Name of the Pinecone index.
            dimension: Dimension of the embeddings.
            force_recreate: Whether to force recreate the index if it exists.
            
        Returns:
            Pinecone index.
        """
        # Initialize Pinecone
        pinecone_client = VectorStore.get_pinecone_client()
        
        # Delete the index if forced recreation
        if force_recreate:
            VectorStore.delete_index(index_name)
        
        # List all indexes
        indexes = pinecone_client.list_indexes()
        
        # Check if the index already exists
        if not any(index.name == index_name for index in indexes):
            # Create the index if it doesn't exist
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created new Pinecone index: {index_name}")
        else:
            print(f"Using existing Pinecone index: {index_name}")
        
        # Return the index
        return pinecone_client.Index(index_name)
    
    @staticmethod
    def create_vector_store(
        documents: List[Document],
        embedding_model_type: str = "bge",
        embedding_model_name: Optional[str] = None,
        index_name: str = "personal-knowledge-assistant"
    ) -> PineconeVectorStore:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to embed.
            embedding_model_type: Type of embeddings model.
            embedding_model_name: Name of the embeddings model.
            index_name: Name of the Pinecone index.
            
        Returns:
            Pinecone vector store.
        """
        # Get embeddings model
        embeddings = EmbeddingGenerator.get_embeddings_model(
            model_type=embedding_model_type,
            model_name=embedding_model_name
        )
        
        # Get Pinecone client
        pinecone_client = VectorStore.get_pinecone_client()
        
        # Delete existing index if it exists (to ensure correct dimensions)
        VectorStore.delete_index(index_name)
        
        # Create or get the index
        # BGE embeddings use 384 dimensions for bge-small-en-v1.5
        dimension = 384
        index = VectorStore.get_or_create_index(index_name, dimension)
        
        # Create the vector store
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
        
        print(f"Documents added to Pinecone index: {index_name}")
        
        return vector_store
    
    @staticmethod
    def get_vector_store(
        embedding_model_type: str = "bge",
        embedding_model_name: Optional[str] = None,
        index_name: str = "personal-knowledge-assistant"
    ) -> PineconeVectorStore:
        """
        Get an existing vector store.
        
        Args:
            embedding_model_type: Type of embeddings model.
            embedding_model_name: Name of the embeddings model.
            index_name: Name of the Pinecone index.
            
        Returns:
            Pinecone vector store.
        """
        # Get embeddings model
        embeddings = EmbeddingGenerator.get_embeddings_model(
            model_type=embedding_model_type,
            model_name=embedding_model_name
        )
        
        # Get Pinecone client
        pinecone_client = VectorStore.get_pinecone_client()
        
        # Get the vector store
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        return vector_store 