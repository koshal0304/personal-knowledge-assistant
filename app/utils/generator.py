from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class Generator:
    """Utility class for generating answers from retrieved context."""
    
    @staticmethod
    def get_llm(model_name: str = "gemini-1.5-pro", temperature: float = 0.0):
        """
        Get a Gemini LLM instance.
        
        Args:
            model_name: Name of the Gemini model.
            temperature: Temperature for generation.
            
        Returns:
            LLM instance.
        """
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    @staticmethod
    def format_documents(docs: List[Document]) -> str:
        """
        Format documents for inclusion in prompt.
        
        Args:
            docs: List of documents.
            
        Returns:
            Formatted string of document contents.
        """
        return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))
    
    @staticmethod
    def build_qa_chain(model_name: str = "gemini-1.5-pro", temperature: float = 0.0):
        """
        Build a question-answering chain using LangChain's updated LCEL structure.
        
        Args:
            model_name: Name of the Gemini model.
            temperature: Temperature for generation.
            
        Returns:
            QA chain.
        """
        # Create the LLM
        llm = Generator.get_llm(model_name, temperature)
        
        # Create the prompt template
        prompt_template = """You are a Personal Knowledge Assistant that provides structured, factual answers based STRICTLY on the provided document context. 

CRITICAL INSTRUCTIONS:
1. ONLY use information that is explicitly present in the provided documents
2. DO NOT add any information, explanations, or details that are not in the documents
3. Present your response in a structured, organized format with clear sections and bullet points where appropriate
4. Use the exact terminology and phrasing from the documents when possible
5. If the documents don't contain information to answer the question, simply state "The documents do not contain information to answer this question" - do not try to be helpful by providing related information
6. Never make up or infer information not present in the documents

Context:
{context}

Question:
{question}

Structured Answer (using ONLY information from the documents):"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the chain using LCEL
        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    
    @staticmethod
    def generate_answer(
        query: str,
        docs: List[Document],
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.1
    ) -> str:
        """
        Generate an answer for a question based on retrieved documents.
        
        Args:
            query: The question to answer.
            docs: Retrieved documents.
            model_name: Name of the Gemini model.
            temperature: Temperature for generation.
            
        Returns:
            Generated answer.
        """
        # Build the QA chain
        chain = Generator.build_qa_chain(model_name, temperature)
        
        # Format the documents
        context = Generator.format_documents(docs)
        
        # Generate the answer
        result = chain.invoke({"context": context, "question": query})
        
        return result 