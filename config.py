"""
Configuration settings for Medical Knowledge RAG Chatbot
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Database Settings
    vector_db_path: str = "./data/vector_db"
    chroma_db_path: str = "./data/chroma_db"
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Data Paths
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    models_dir: str = "./data/models"
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_length: int = 512
    
    # RAG Configuration
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.data_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.models_dir,
        settings.vector_db_path,
        settings.chroma_db_path,
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()