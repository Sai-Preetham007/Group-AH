#!/usr/bin/env python3
"""
Script to run the Medical Knowledge RAG Chatbot API server
"""
import uvicorn
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True
    )
