import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from utils.document_loaders import DocumentLoader
from utils.preprocessor import DocumentPreprocessor
from utils.vector_store import VectorStore

# Load environment variables
load_dotenv()

def ingest_documents(
    data_dir: str,
    embedding_model_type: str = "bge",
    embedding_model_name: str = None,
    index_name: str = "personal-knowledge-assistant",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    Ingest documents into the vector store.
    
    Args:
        data_dir: Directory containing documents.
        embedding_model_type: Type of embeddings model.
        embedding_model_name: Name of the embeddings model.
        index_name: Name of the Pinecone index.
        chunk_size: Size of each chunk in tokens.
        chunk_overlap: Number of tokens to overlap between chunks.
    """
    print(f"Loading documents from {data_dir}...")
    documents = DocumentLoader.load_from_directory(data_dir)
    print(f"Loaded {len(documents)} documents.")
    
    print("Preprocessing and chunking documents...")
    chunked_documents = DocumentPreprocessor.chunk_documents(
        documents, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    print(f"Created {len(chunked_documents)} chunks.")
    
    print(f"Creating vector store with {embedding_model_type} embeddings...")
    vector_store = VectorStore.create_vector_store(
        chunked_documents,
        embedding_model_type=embedding_model_type,
        embedding_model_name=embedding_model_name,
        index_name=index_name
    )
    print(f"Vector store created with index name: {index_name}")
    
    return vector_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into a vector store.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing documents")
    parser.add_argument("--embedding-model-type", type=str, default="bge", choices=["bge", "huggingface"], help="Type of embeddings model")
    parser.add_argument("--embedding-model-name", type=str, default=None, help="Name of the embeddings model")
    parser.add_argument("--index-name", type=str, default="personal-knowledge-assistant", help="Name of the Pinecone index")
    parser.add_argument("--chunk-size", type=int, default=512, help="Size of each chunk in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Number of tokens to overlap between chunks")
    
    args = parser.parse_args()
    
    ingest_documents(
        data_dir=args.data_dir,
        embedding_model_type=args.embedding_model_type,
        embedding_model_name=args.embedding_model_name,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    ) 