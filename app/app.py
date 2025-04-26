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
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700&display=swap');
    
        /* Main app styling with dark cyberpunk theme */
        .main {
            background-color: #0d1117;
            font-family: 'Outfit', sans-serif;
            color: #e6edf3;
            background-image: 
                radial-gradient(circle at 25px 25px, rgba(0, 255, 170, 0.05) 2%, transparent 0%),
                radial-gradient(circle at 75px 75px, rgba(255, 0, 128, 0.05) 2%, transparent 0%);
            background-size: 100px 100px;
            animation: backgroundShift 120s linear infinite;
        }
        
        @keyframes backgroundShift {
            0% { background-position: 0 0; }
            100% { background-position: 100px 100px; }
        }
        
        /* Neon glow effects */
        .neon-cyan {
            text-shadow: 0 0 5px rgba(0, 255, 255, 0.7), 0 0 10px rgba(0, 255, 255, 0.5), 0 0 20px rgba(0, 255, 255, 0.3);
            color: #00ffff;
        }
        
        .neon-magenta {
            text-shadow: 0 0 5px rgba(255, 0, 128, 0.7), 0 0 10px rgba(255, 0, 128, 0.5), 0 0 20px rgba(255, 0, 128, 0.3);
            color: #ff0080;
        }
        
        /* Glass morphism effect for containers - dark version */
        .glass-effect {
            background: rgba(20, 29, 47, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        /* Custom sidebar styling with neon border */
        .css-1cypcdb, .css-81oif8 {
            background-color: #0d1117;
            color: #e6edf3;
            border-right: 1px solid rgba(0, 255, 170, 0.3);
            box-shadow: 5px 0 15px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }
        
        .css-1cypcdb::before, .css-81oif8::before {
            content: "";
            position: absolute;
            top: 0;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 255, 170, 0.1) 0%, transparent 60%);
            animation: sidebarglow 15s ease-in-out infinite alternate;
            z-index: 0;
        }
        
        @keyframes sidebarglow {
            0% { transform: translateY(-30%) rotate(0deg); opacity: 0.3; }
            100% { transform: translateY(-10%) rotate(10deg); opacity: 0.5; }
        }
        
        /* Header styling - cyberpunk version */
        .main-header {
            background: linear-gradient(125deg, #131b29 0%, #0d1117 100%);
            color: #e6edf3;
            padding: 3rem 2.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            text-align: center;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(0, 255, 170, 0.2);
            transition: all 0.5s ease;
        }
        
        .main-header:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 255, 170, 0.2);
            border: 1px solid rgba(0, 255, 170, 0.4);
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(0, 255, 170, 0.1) 0%, transparent 70%);
            z-index: 0;
        }
        
        .main-header::after {
            content: "";
            position: absolute;
            height: 2px;
            width: 60%;
            left: 20%;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 170, 0.8), transparent);
            z-index: 1;
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .main-header h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            letter-spacing: 1px;
            animation: textGlow 3s ease-in-out infinite alternate;
        }
        
        @keyframes textGlow {
            0% {
                color: #ffffff;
                text-shadow: 0 0 5px rgba(0, 255, 170, 0.5),
                             0 0 10px rgba(0, 255, 170, 0.3);
            }
            100% {
                color: #ffffff;
                text-shadow: 0 0 10px rgba(0, 255, 170, 0.7),
                             0 0 20px rgba(0, 255, 170, 0.5),
                             0 0 30px rgba(0, 255, 170, 0.3);
            }
        }
        
        .main-header p {
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
            letter-spacing: 0.5px;
            animation: fadeIn 1s ease-in-out;
            color: #c9d1d9;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 0.9; transform: translateY(0); }
        }
        
        .header-highlight {
            color: #ff0080;
            font-weight: 500;
            position: relative;
            animation: pulse 3s infinite;
        }
        
        @keyframes pulse {
            0% { text-shadow: 0 0 0 rgba(255, 0, 128, 0); }
            50% { text-shadow: 0 0 10px rgba(255, 0, 128, 0.7), 0 0 20px rgba(255, 0, 128, 0.5); }
            100% { text-shadow: 0 0 0 rgba(255, 0, 128, 0); }
        }
        
        /* Header decorative elements */
        .header-dots {
            position: absolute;
            bottom: -20px;
            right: -20px;
            width: 150px;
            height: 150px;
            background: url('data:image/svg+xml,<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><circle cx="2" cy="2" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="10" cy="2" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="18" cy="2" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="2" cy="10" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="10" cy="10" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="18" cy="10" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="2" cy="18" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="10" cy="18" r="1" fill="%2300ffaa" opacity="0.5" /><circle cx="18" cy="18" r="1" fill="%2300ffaa" opacity="0.5" /></svg>');
            background-size: 20px 20px;
            z-index: 0;
            opacity: 0.3;
            animation: dotsFloat 20s linear infinite;
        }
        
        @keyframes dotsFloat {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .header-graphic {
            position: absolute;
            top: -30px;
            left: -30px;
            width: 120px;
            height: 120px;
            opacity: 0.3;
            background: radial-gradient(circle, rgba(255, 0, 128, 0.8) 0%, rgba(255, 0, 128, 0.3) 50%, rgba(0, 0, 0, 0) 75%);
            border-radius: 50%;
            z-index: 0;
            animation: graphicPulse 10s ease-in-out infinite alternate;
        }
        
        @keyframes graphicPulse {
            0% { transform: scale(1); opacity: 0.2; }
            50% { transform: scale(1.2); opacity: 0.3; }
            100% { transform: scale(1); opacity: 0.2; }
        }
        
        /* Cards styling with neon borders */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            background-color: #141d2f;
            border: 1px solid rgba(0, 255, 170, 0.2);
            overflow: hidden;
        }
        
        .stCard:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 255, 170, 0.3);
            border: 1px solid rgba(0, 255, 170, 0.5);
        }
        
        /* Chat container with dark glass effect */
        .chat-container {
            border-radius: 10px;
            padding: 1.5rem;
            background-color: #141d2f;
            border: 1px solid rgba(0, 255, 170, 0.2);
            max-height: 600px;
            overflow-y: auto;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            scrollbar-width: thin;
            scrollbar-color: #00ffaa transparent;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 4px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background-color: rgba(0, 255, 170, 0.5);
            border-radius: 20px;
        }
        
        .chat-container:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 255, 170, 0.1);
            border: 1px solid rgba(0, 255, 170, 0.3);
        }
        
        /* User message styling with neon accents */
        .user-message {
            background-color: #1c2333;
            color: #e6edf3;
            border-radius: 10px 10px 0 10px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 5px rgba(255, 0, 128, 0.1);
            animation: slideInRight 0.3s ease-out forwards;
            position: relative;
            transform-origin: top right;
            border-left: 2px solid #ff0080;
        }
        
        @keyframes slideInRight {
            from { transform: translateX(20px) scale(0.95); opacity: 0; }
            to { transform: translateX(0) scale(1); opacity: 1; }
        }
        
        .user-message::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(to left, #ff0080, transparent);
            animation: userMessageBorderGlow 2s ease-out forwards;
        }
        
        @keyframes userMessageBorderGlow {
            from { width: 0; }
            to { width: 100%; }
        }
        
        .user-message::after {
            content: "";
            position: absolute;
            right: 0;
            bottom: -6px;
            width: 15px;
            height: 15px;
            background: linear-gradient(135deg, #1c2333 0%, #1c2333 50%, transparent 50%, transparent 100%);
            transform: rotate(45deg);
        }
        
        /* Assistant message styling with green neon accents */
        .assistant-message {
            background-color: #192231;
            color: #e6edf3;
            border-radius: 10px 10px 10px 0;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 5px rgba(0, 255, 170, 0.1);
            animation: slideInLeft 0.3s ease-out forwards;
            position: relative;
            transform-origin: top left;
            border-right: 2px solid #00ffaa;
        }
        
        @keyframes slideInLeft {
            from { transform: translateX(-20px) scale(0.95); opacity: 0; }
            to { transform: translateX(0) scale(1); opacity: 1; }
        }
        
        .assistant-message::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(to right, #00ffaa, transparent);
            animation: assistantMessageBorderGlow 2s ease-out forwards;
        }
        
        @keyframes assistantMessageBorderGlow {
            from { width: 0; }
            to { width: 100%; }
        }
        
        .assistant-message::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -6px;
            width: 15px;
            height: 15px;
            background: linear-gradient(135deg, transparent 0%, transparent 50%, #192231 50%, #192231 100%);
            transform: rotate(45deg);
        }
        
        /* Terminal-style input box */
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 0.8rem 1.5rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1), 0 0 5px rgba(0, 255, 170, 0.1);
            border: 1px solid rgba(0, 255, 170, 0.3);
            background-color: #0d1117;
            color: #e6edf3;
            font-family: 'Outfit', sans-serif;
            transition: all 0.3s ease;
            letter-spacing: 0.5px;
        }
        
        .stTextInput>div>div>input:focus {
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 255, 170, 0.2);
            border: 1px solid rgba(0, 255, 170, 0.6);
            outline: none;
        }
        
        .stTextInput>div>div>input::placeholder {
            color: rgba(230, 237, 243, 0.5);
        }
        
        /* Cyberpunk sidebar headers */
        .sidebar-header {
            background-color: #192231;
            padding: 0.8rem;
            border-radius: 5px;
            margin-top: 1.5rem;
            color: #e6edf3;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
            border-left: 2px solid #00ffaa;
        }
        
        .sidebar-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(to right, #00ffaa, transparent);
            animation: borderGlow 2s ease-out forwards;
        }
        
        @keyframes borderGlow {
            from { width: 0; }
            to { width: 100%; }
        }
        
        /* Futuristic upload container */
        .upload-container {
            border: 1px dashed rgba(0, 255, 170, 0.3);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
            background-color: rgba(20, 29, 47, 0.7);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }
        
        .upload-container::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 170, 0.1), transparent);
            transition: all 0.5s ease;
        }
        
        .upload-container:hover {
            border-color: #00ffaa;
            background-color: rgba(25, 34, 49, 0.8);
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), 0 0 15px rgba(0, 255, 170, 0.1);
        }
        
        .upload-container:hover::before {
            left: 100%;
        }
        
        /* Neon button styling */
        .stButton>button {
            background-color: #192231;
            color: #00ffaa;
            border-radius: 5px;
            border: 1px solid rgba(0, 255, 170, 0.3);
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            letter-spacing: 1px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            z-index: 1;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2), 0 0 10px rgba(0, 255, 170, 0.1);
            text-shadow: 0 0 5px rgba(0, 255, 170, 0.5);
        }
        
        .stButton>button::after {
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            width: 150%;
            height: 150%;
            background: radial-gradient(circle, rgba(0, 255, 170, 0.2) 0%, transparent 70%);
            transform: translate(-50%, -50%) scale(0);
            opacity: 0;
            z-index: -1;
            transition: transform 0.5s, opacity 0.5s;
            pointer-events: none;
        }
        
        .stButton>button:hover {
            color: #ffffff;
            background-color: #192231;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 0 0 15px rgba(0, 255, 170, 0.3);
            text-shadow: 0 0 10px rgba(0, 255, 170, 0.7);
            border-color: rgba(0, 255, 170, 0.7);
        }
        
        .stButton>button:hover::after {
            transform: translate(-50%, -50%) scale(1);
            opacity: 1;
        }
        
        /* Typing indicator with neon dots */
        .typing-indicator {
            display: flex;
            align-items: center;
            margin: 0.8rem 0;
            padding: 0.5rem;
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            margin: 0 3px;
            background-color: #00ffaa;
            border-radius: 50%;
            display: inline-block;
            animation: cyberpunkBounce 1.5s cubic-bezier(0.42, 0, 0.58, 1) infinite;
            box-shadow: 0 0 5px rgba(0, 255, 170, 0.7), 0 0 10px rgba(0, 255, 170, 0.3);
        }
        
        .typing-indicator span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes cyberpunkBounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        
        /* Logo styling with neon effect */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .logo-container span {
            font-size: 2.5rem;
            animation: brainPulse 5s ease-in-out infinite;
            filter: drop-shadow(0 0 10px rgba(0, 255, 170, 0.7));
            color: #00ffaa;
        }
        
        @keyframes brainPulse {
            0%, 100% { transform: scale(1); filter: drop-shadow(0 0 5px rgba(0, 255, 170, 0.7)); }
            50% { transform: scale(1.1); filter: drop-shadow(0 0 15px rgba(0, 255, 170, 0.9)); }
        }
        
        .logo-text {
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            font-size: 1.6rem;
            margin-left: 0.7rem;
            color: #ffffff;
            letter-spacing: 2px;
            text-shadow: 0 0 5px rgba(0, 255, 170, 0.7), 0 0 10px rgba(0, 255, 170, 0.5);
        }
        
        /* Cybertech loading animation */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 120px;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 2px solid transparent;
            border-top: 2px solid #00ffaa;
            border-right: 2px solid transparent;
            border-radius: 50%;
            animation: cyberspin 1s linear infinite;
            position: relative;
        }
        
        .loading-spinner::before, .loading-spinner::after {
            content: "";
            position: absolute;
            top: -2px;
            left: -2px;
            width: 100%;
            height: 100%;
            border: 2px solid transparent;
            border-top: 2px solid #ff0080;
            border-radius: 50%;
            animation: cyberspin 2s linear infinite reverse;
        }
        
        .loading-spinner::after {
            width: 70%;
            height: 70%;
            top: 13%;
            left: 13%;
            border-top: 2px solid #00ffaa;
            animation: cyberspin 1.5s linear infinite;
        }
        
        @keyframes cyberspin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Matrix raining code effect */
        .matrix-code {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.05;
            pointer-events: none;
            overflow: hidden;
        }
        
        /* Progress bar enhancements */
        .stProgress > div > div > div {
            background-color: #00ffaa !important;
            box-shadow: 0 0 10px rgba(0, 255, 170, 0.7), 0 0 5px rgba(0, 255, 170, 0.5);
        }
    </style>
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Matrix code effect script -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Create matrix container
            const container = document.createElement('div');
            container.className = 'matrix-code';
            document.body.appendChild(container);
            
            // Set up canvas
            const canvas = document.createElement('canvas');
            container.appendChild(canvas);
            
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            // Matrix characters
            const chars = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン";
            
            const columns = Math.floor(canvas.width / 15);
            const drops = [];
            
            // Initialize drops
            for (let i = 0; i < columns; i++) {
                drops[i] = Math.floor(Math.random() * -100);
            }
            
            function draw() {
                // Semi-transparent black to create fade effect
                ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Green text
                ctx.fillStyle = "#00ffaa";
                ctx.font = "15px monospace";
                
                // Draw characters
                for (let i = 0; i < drops.length; i++) {
                    // Random character
                    const text = chars[Math.floor(Math.random() * chars.length)];
                    
                    // x = i * font width, y = drop position
                    ctx.fillText(text, i * 15, drops[i] * 15);
                    
                    // Reset drop if it reaches bottom or random chance
                    if (drops[i] * 15 > canvas.height && Math.random() > 0.99) {
                        drops[i] = 0;
                    }
                    
                    // Move drop down
                    drops[i]++;
                }
                
                // Call function again
                requestAnimationFrame(draw);
            }
            
            // Start animation
            draw();
            
            // Resize handler
            window.addEventListener('resize', function() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            });
        });
    </script>
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

if "sample_shown" not in st.session_state:
    st.session_state.sample_shown = False

# Sidebar for settings and file upload
with st.sidebar:
    st.markdown('<div class="logo-container"><span style="font-size:30px;">🧠</span><div class="logo-text">NEURO<span style="color:#00ffaa">PKA</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">SYSTEM CONFIG</div>', unsafe_allow_html=True)
    
    # Model settings
    st.subheader("Neural Network")
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
        options=["gemini-1.5-pro"],
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
                    
                    # Show sample format if first time loading documents
                    if not st.session_state.sample_shown:
                        st.session_state.messages = [
                            {"role": "assistant", "content": """**Welcome to your Personal Knowledge Assistant!**

I'll provide structured answers using only information from your documents. Here's an example of the structured response format:

**Topic: [Main Topic]**
- Key information point 1 from documents
- Key information point 2 from documents

**Details:**
• Specific detail 1 with exact terminology from documents
• Specific detail 2 with exact terminology from documents

**References:**
• Information sourced from Document 1
• Information sourced from Document 2

You can ask questions like "What are the key features of [topic]?" or "Explain the process of [topic]" and I'll provide organized answers using only information contained in your documents."""}
                        ]
                        st.session_state.sample_shown = True
                    
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                st.session_state.processing = False

# Main content
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    # Custom header with cyberpunk styling
    st.markdown('''
    <div class="main-header">
        <div class="header-graphic"></div>
        <div class="header-dots"></div>
        <div class="header-content">
            <h1>NEURO<span style="color:#00ffaa">PKA</span></h1>
            <p>Access your <span class="header-highlight">neural knowledge matrix</span> with advanced AI</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display introduction if no vector store
    if not st.session_state.vector_store:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #141d2f; border-radius: 10px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 255, 170, 0.2);">
            <div style="font-size: 64px; margin-bottom: 1rem; text-shadow: 0 0 10px rgba(0, 255, 170, 0.7), 0 0 20px rgba(0, 255, 170, 0.3);">🧠</div>
            <h2 style="font-family: 'Orbitron', sans-serif; color: #e6edf3; margin-bottom: 1rem; text-shadow: 0 0 5px rgba(0, 255, 170, 0.5);">NEURAL INTERFACE READY</h2>
            <p style="color: #c9d1d9; margin-bottom: 2rem;">Upload your data files to the neural network for cognitive analysis</p>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem;">
                <div style="background-color: #192231; border-radius: 10px; padding: 1.5rem; width: 200px; text-align: center; border-left: 2px solid #00ffaa; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);">
                    <h3 style="color: #00ffaa; font-family: 'Orbitron', sans-serif;">01</h3>
                    <h4 style="color: #e6edf3;">UPLOAD</h4>
                    <p style="color: #a0aec0; font-size: 0.9rem;">Upload data files to neural core</p>
                </div>
                <div style="background-color: #192231; border-radius: 10px; padding: 1.5rem; width: 200px; text-align: center; border-left: 2px solid #00ffaa; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);">
                    <h3 style="color: #00ffaa; font-family: 'Orbitron', sans-serif;">02</h3>
                    <h4 style="color: #e6edf3;">PROCESS</h4>
                    <p style="color: #a0aec0; font-size: 0.9rem;">Neural embedding activation</p>
                </div>
                <div style="background-color: #192231; border-radius: 10px; padding: 1.5rem; width: 200px; text-align: center; border-left: 2px solid #00ffaa; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);">
                    <h3 style="color: #00ffaa; font-family: 'Orbitron', sans-serif;">03</h3>
                    <h4 style="color: #e6edf3;">INTERFACE</h4>
                    <p style="color: #a0aec0; font-size: 0.9rem;">Neural data query interface</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat interface with improved styling
        st.markdown('<h2 style="font-family: \'Orbitron\', sans-serif; color: #e6edf3; margin-bottom: 1rem; text-shadow: 0 0 5px rgba(0, 255, 170, 0.5);">NEURAL INTERFACE</h2>', unsafe_allow_html=True)
        
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