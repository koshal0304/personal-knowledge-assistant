import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import time

from utils.document_loaders import DocumentLoader
from utils.preprocessor import DocumentPreprocessor
from utils.vector_store import VectorStore
from utils.retriever import Retriever
from utils.generator import Generator

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for a more attractive UI
def local_css():
    css = """
    <style>
        /* Main app styling */
        .main {
            background-color: #f5f7ff;
            font-family: 'Inter', sans-serif;
        }
        
        /* Custom sidebar styling */
        .css-1cypcdb {
            background-color: #1e293b;
            color: white;
            border-right: 1px solid #334155;
        }
        
        /* Header styling - modern version */
        .main-header {
            background: linear-gradient(125deg, #2d3748 0%, #1e293b 40%, #0f172a 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 15px 35px rgba(15, 23, 42, 0.2), 0 5px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml,<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L100 0 L100 100 L0 100 Z" fill="none" stroke="white" stroke-opacity="0.03" stroke-width="1" /></svg>'), 
                       url('data:image/svg+xml,<svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40" fill="none" stroke="white" stroke-opacity="0.03" stroke-width="1" /></svg>');
            background-position: 0 0, 40px 40px;
            background-size: 100px 100px, 80px 80px;
            z-index: 0;
            opacity: 0.7;
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .main-header h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            color: white;
            text-shadow: 0 2px 5px rgba(0,0,0,0.3);
            letter-spacing: -0.02em;
        }
        
        .main-header p {
            font-size: 1.2rem;
            font-weight: 300;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
            letter-spacing: 0.5px;
        }
        
        .header-highlight {
            color: #d8b4fe;
            font-weight: 500;
        }
        
        .header-dots {
            position: absolute;
            bottom: -20px;
            right: -20px;
            width: 150px;
            height: 150px;
            background: url('data:image/svg+xml,<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><circle cx="2" cy="2" r="1" fill="white" opacity="0.3" /><circle cx="10" cy="2" r="1" fill="white" opacity="0.3" /><circle cx="18" cy="2" r="1" fill="white" opacity="0.3" /><circle cx="2" cy="10" r="1" fill="white" opacity="0.3" /><circle cx="10" cy="10" r="1" fill="white" opacity="0.3" /><circle cx="18" cy="10" r="1" fill="white" opacity="0.3" /><circle cx="2" cy="18" r="1" fill="white" opacity="0.3" /><circle cx="10" cy="18" r="1" fill="white" opacity="0.3" /><circle cx="18" cy="18" r="1" fill="white" opacity="0.3" /></svg>');
            background-size: 20px 20px;
            z-index: 0;
            opacity: 0.3;
        }
        
        .header-graphic {
            position: absolute;
            top: -30px;
            left: -30px;
            width: 120px;
            height: 120px;
            opacity: 0.15;
            background: radial-gradient(circle, rgba(216, 180, 254, 0.8) 0%, rgba(107, 70, 193, 0.3) 50%, rgba(0, 0, 0, 0) 75%);
            border-radius: 50%;
            z-index: 0;
        }
        
        /* Cards styling */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .stCard:hover {
            transform: translateY(-5px);
        }
        
        /* Chat container */
        .chat-container {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            background-color: white;
            max-height: 600px;
            overflow-y: auto;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* User message styling */
        .user-message {
            background-color: #e2e8f0;
            color: #1e293b;
            border-radius: 20px 20px 0 20px;
            padding: 0.8rem 1.2rem;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
        }
        
        /* Assistant message styling */
        .assistant-message {
            background-color: #7c3aed;
            color: white;
            border-radius: 20px 20px 20px 0;
            padding: 0.8rem 1.2rem;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 2px 4px rgba(124, 58, 237, 0.3);
        }
        
        /* Prompt box styling */
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding-left: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Sidebar headers */
        .sidebar-header {
            background-color: #334155;
            padding: 0.5rem;
            border-radius: 5px;
            margin-top: 1rem;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        
        /* Upload container */
        .upload-container {
            border: 2px dashed #cbd5e1;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
            background-color: #f8fafc;
            transition: all 0.3s ease;
        }
        .upload-container:hover {
            border-color: #7c3aed;
            background-color: #f1f5f9;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #7c3aed;
            color: white;
            border-radius: 25px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #6d28d9;
            box-shadow: 0 4px 6px rgba(109, 40, 217, 0.3);
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: #7c3aed;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
            animation: bounce 1.3s linear infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.15s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.3s;
        }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        /* Logo styling */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }
        .logo-text {
            font-weight: bold;
            font-size: 1.2rem;
            margin-left: 0.5rem;
            color: #7c3aed;
        }
        
        /* Loading animation */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #7c3aed;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply custom CSS
local_css()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Sidebar for settings and file upload
with st.sidebar:
    st.markdown('<div class="logo-container"><span style="font-size:30px;">🧠</span><div class="logo-text">PKA</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Models & Configuration</div>', unsafe_allow_html=True)
    
    # Model settings
    st.subheader("Embedding Model")
    embedding_model_type = st.selectbox(
        "Select model",
        options=["bge"],
        index=0,
        help="Choose the embedding model for vectorizing your documents"
    )
    
    embedding_model_name = None
    if embedding_model_type == "bge":
        embedding_model_name = "BAAI/bge-small-en-v1.5"
    
    st.subheader("LLM Model")
    llm_model = st.selectbox(
        "Select model",
        options=["gemini-pro", "gemini-1.5-pro"],
        index=0,
        help="Choose the language model for generating answers"
    )
    
    # Vector store settings
    st.markdown('<div class="sidebar-header">Vector Store</div>', unsafe_allow_html=True)
    index_name = st.text_input("Index Name", "personal-knowledge-assistant", help="Name of your Pinecone vector index")
    recreate_index = st.checkbox("Recreate Index", False, help="Delete and recreate the index even if it exists")
    
    # Chunking settings
    st.markdown('<div class="sidebar-header">Document Processing</div>', unsafe_allow_html=True)
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64, help="Size of each document chunk in characters")
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 50, 8, help="Overlap between consecutive chunks")
    
    # Retrieval settings
    st.subheader("Retrieval")
    num_results = st.slider("Number of Results", 1, 10, 5, 1, help="Number of document chunks to retrieve per query")
    use_compression = st.checkbox("Use Context Compression", False, help="Enable to filter and compress retrieved chunks")
    
    # File upload section
    st.markdown('<div class="sidebar-header">Document Upload</div>', unsafe_allow_html=True)
    
    # Animated upload container
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop your files here", accept_multiple_files=True, 
                                      help="Upload PDF, Markdown, or text files")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        process_button = st.button("Process Documents", help="Click to process and index uploaded documents")
        
        if process_button:
            st.session_state.processing = True
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to temp directory
                    with st.spinner("Saving uploaded files..."):
                        for file in uploaded_files:
                            file_path = os.path.join(temp_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                    
                    # Process the documents with step-by-step progress
                    progress_bar = st.progress(0)
                    
                    # Step 1: Load documents
                    st.info("Step 1/4: Loading documents...")
                    documents = DocumentLoader.load_from_directory(temp_dir)
                    st.success(f"✅ Loaded {len(documents)} documents")
                    progress_bar.progress(25)
                    
                    # Step 2: Preprocess and chunk
                    st.info("Step 2/4: Preprocessing and chunking...")
                    chunked_documents = DocumentPreprocessor.chunk_documents(
                        documents, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    st.success(f"✅ Created {len(chunked_documents)} chunks")
                    progress_bar.progress(50)
                    
                    # Step 3: Delete existing index if requested
                    if recreate_index:
                        st.info("Step 3/4: Recreating vector index...")
                        VectorStore.delete_index(index_name)
                        st.success("✅ Deleted existing index")
                    else:
                        st.info("Step 3/4: Setting up vector store...")
                    progress_bar.progress(75)
                    
                    # Step 4: Create vector store
                    st.info("Step 4/4: Creating vector embeddings...")
                    vector_store = VectorStore.create_vector_store(
                        chunked_documents,
                        embedding_model_type=embedding_model_type,
                        embedding_model_name=embedding_model_name,
                        index_name=index_name
                    )
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    
                    # Create the retriever
                    retriever = Retriever.build_hybrid_retriever(
                        vector_store,
                        k=num_results,
                        use_compression=use_compression,
                        llm_model_name=llm_model
                    )
                    
                    # Store the retriever in session state
                    st.session_state.retriever = retriever
                    
                    progress_bar.progress(100)
                    st.success("✅ Documents processed successfully!")
                    time.sleep(1)  # Allow user to see the success message
                    progress_bar.empty()
                    
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                st.session_state.processing = False

# Main content
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    # Custom header with solid white text instead of gradient
    st.markdown('''
    <div class="main-header">
        <div class="header-graphic"></div>
        <div class="header-dots"></div>
        <div class="header-content">
            <h1>Personal Knowledge Assistant</h1>
            <p>Upload documents and chat with your <span class="header-highlight">knowledge base</span></p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display introduction if no vector store
    if not st.session_state.vector_store:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <img src="https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/balloons.jpg" width="200" style="border-radius: 10px; margin-bottom: 1rem;">
            <h2>Welcome to your Personal Knowledge Assistant</h2>
            <p>This assistant helps you manage and query your knowledge base using Google's Gemini AI.</p>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem;">
                <div style="background-color: #f8fafc; border-radius: 10px; padding: 1rem; width: 200px; text-align: center;">
                    <h3>📄</h3>
                    <h4>Upload Documents</h4>
                    <p>Upload PDF, Markdown, and text files</p>
                </div>
                <div style="background-color: #f8fafc; border-radius: 10px; padding: 1rem; width: 200px; text-align: center;">
                    <h3>🔍</h3>
                    <h4>Process Content</h4>
                    <p>Automatically chunk and embed your content</p>
                </div>
                <div style="background-color: #f8fafc; border-radius: 10px; padding: 1rem; width: 200px; text-align: center;">
                    <h3>💬</h3>
                    <h4>Chat & Search</h4>
                    <p>Ask questions in natural language</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat interface with improved styling
        st.markdown('<h2>Chat with your documents</h2>', unsafe_allow_html=True)
        
        # Chat container
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom of chat (JavaScript)
        st.markdown("""
        <script>
            function scrollToBottom() {
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            scrollToBottom();
        </script>
        """, unsafe_allow_html=True)
        
        # Define callback functions
        def process_query():
            if st.session_state.user_input and not st.session_state.waiting_for_answer:
                user_query = st.session_state.user_input
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.user_input = ""
                st.session_state.waiting_for_answer = True
                st.rerun()
        
        # Input for user query with better styling
        user_query = st.text_input(
            "Ask a question about your documents",
            key="user_input",
            placeholder="Type your question here...",
            on_change=process_query
        )
        
        # Process the new message if waiting for answer
        if st.session_state.waiting_for_answer and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_query = st.session_state.messages[-1]["content"]
            
            # Display typing indicator
            with st.spinner(""):
                st.markdown(
                    """
                    <div class="assistant-message" style="width: auto; padding: 10px;">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                try:
                    # Retrieve relevant documents
                    docs = Retriever.retrieve_documents(
                        user_query,
                        st.session_state.retriever
                    )
                    
                    # Generate answer
                    answer = Generator.generate_answer(
                        user_query,
                        docs,
                        model_name=llm_model
                    )
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                # Reset waiting flag
                st.session_state.waiting_for_answer = False
                st.rerun()
        
        # Add a clear chat button
        if st.session_state.messages:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.waiting_for_answer = False
                st.rerun() 