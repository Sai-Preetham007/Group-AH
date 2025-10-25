"""
Logging utilities for Medical Knowledge RAG Chatbot
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import settings


def setup_logger(name: str = "medical_rag", 
                log_file: Optional[str] = None,
                level: str = None) -> logging.Logger:
    """Setup logger for the application"""
    
    # Get log level from settings or parameter
    log_level = level or settings.log_level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "medical_rag") -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name)
