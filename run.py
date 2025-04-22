#!/usr/bin/env python3
"""
Launcher script for the Personal Knowledge Assistant.
"""
import os
import sys
import subprocess
import shutil

def main():
    """Launch the Personal Knowledge Assistant."""
    # Check if the .env file exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Creating a template...")
        shutil.copy(".env.example", ".env")
        print("Please edit the .env file with your API keys before running the application.")
        sys.exit(1)
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "app/app.py"])

if __name__ == "__main__":
    main() 