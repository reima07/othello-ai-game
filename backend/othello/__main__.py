#!/usr/bin/env python3
"""
Othello AI Engine - Main Entry Point
"""

import sys
import os

# Add the parent directory to the path so we can import othello modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from othello.server import app
import uvicorn

if __name__ == "__main__":
    print("Starting Othello AI Engine...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
