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
    def get_llm(model_name: str = "gemini-pro", temperature: float = 0.0):
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
    def build_qa_chain(model_name: str = "gemini-pro", temperature: float = 0.0):
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
        prompt_template = """You are a Personal Knowledge Assistant with expertise in analyzing and explaining documents. Your goal is to provide comprehensive, helpful answers based on the provided context. 

When a user asks a question:
1. Analyze the context thoroughly and extract all relevant information
2. Provide detailed, informative responses that directly answer the question
3. If the answer is partially in the context, provide what you know and indicate any gaps
4. If asked to explain a concept mentioned in the context, elaborate on it using both the context information and your general knowledge
5. Always try to be helpful - avoid saying "I don't know" or asking for clarification unless absolutely necessary

Context:
{context}

Question:
{question}

Answer (be thorough and detailed):"""
        
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
        model_name: str = "gemini-pro",
        temperature: float = 0.3
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