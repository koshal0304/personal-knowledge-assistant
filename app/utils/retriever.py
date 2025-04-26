from typing import List, Dict, Any, Optional
import re
import os

from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI

from .vector_store import VectorStore

class Retriever:
    """Utility class for retrieving relevant documents."""
    
    @staticmethod
    def build_hybrid_retriever(
        vector_store,
        k: int = 5,
        use_compression: bool = False,
        llm_model_name: str = "gemini-1.5-pro"
    ):
        """
        Build a hybrid retriever that combines semantic and keyword search.
        
        Args:
            vector_store: The vector store to use for retrieval.
            k: Number of documents to retrieve.
            use_compression: Whether to use contextual compression.
            llm_model_name: Name of the LLM to use for compression.
            
        Returns:
            A hybrid retriever.
        """
        # Create a base retriever
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        if not use_compression:
            return base_retriever
        
        # Create an LLM for compression
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create a compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create a compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    @staticmethod
    def retrieve_documents(
        query: str,
        retriever,
        max_tokens: int = 4000
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The query string.
            retriever: The retriever to use.
            max_tokens: Maximum number of tokens to retrieve.
            
        Returns:
            List of retrieved documents.
        """
        # Retrieve documents using the newer invoke method
        docs = retriever.invoke(query)
        
        # Count tokens (approximate method)
        def count_tokens(text):
            # Approximate token count (average 4 chars per token)
            return len(text) // 4
        
        # Filter documents to stay within token limit
        filtered_docs = []
        current_tokens = 0
        
        for i, doc in enumerate(docs):
            doc_tokens = count_tokens(doc.page_content)
            
            if current_tokens + doc_tokens <= max_tokens:
                # Add source information to document metadata
                if 'source' in doc.metadata:
                    source_info = doc.metadata['source']
                else:
                    source_info = f"Document {i+1}"
                
                # Enrich the document content with its source info
                doc.page_content = f"{doc.page_content}\n\nSource: {source_info}"
                
                filtered_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break
        
        return filtered_docs 