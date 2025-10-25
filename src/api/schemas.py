"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request schema for medical queries"""
    question: str = Field(..., description="Medical question to answer", min_length=10, max_length=500)
    top_k: Optional[int] = Field(default=5, description="Number of relevant documents to retrieve", ge=1, le=20)
    include_sources: Optional[bool] = Field(default=True, description="Include source citations")
    safety_check: Optional[bool] = Field(default=True, description="Enable safety validation")


class QueryResponse(BaseModel):
    """Response schema for medical queries"""
    response: str = Field(..., description="Generated medical response")
    sources: List[Dict[str, str]] = Field(..., description="Source citations")
    context_used: int = Field(..., description="Number of context documents used")
    query: str = Field(..., description="Original query")
    confidence: Optional[float] = Field(default=None, description="Response confidence score")
    safety_score: Optional[float] = Field(default=None, description="Safety validation score")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="System status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    vector_store_stats: Optional[Dict[str, Any]] = Field(default=None, description="Vector store statistics")
    model_info: Optional[Dict[str, str]] = Field(default=None, description="Model information")


class StatsResponse(BaseModel):
    """Statistics response schema"""
    vector_store_stats: Dict[str, Any] = Field(..., description="Vector store statistics")
    model_info: Dict[str, str] = Field(..., description="Model information")
    system_info: Dict[str, Any] = Field(..., description="System information")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")


class ValidationRequest(BaseModel):
    """Request schema for content validation"""
    text: str = Field(..., description="Text to validate", min_length=1, max_length=1000)
    validation_type: str = Field(default="medical", description="Type of validation to perform")


class ValidationResponse(BaseModel):
    """Response schema for content validation"""
    is_safe: bool = Field(..., description="Whether content is safe")
    confidence: float = Field(..., description="Validation confidence score")
    warnings: List[str] = Field(..., description="Safety warnings")
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    risk_level: str = Field(..., description="Risk level assessment")


class TrainingRequest(BaseModel):
    """Request schema for model training"""
    dataset_path: str = Field(..., description="Path to training dataset")
    model_name: str = Field(default="microsoft/DialoGPT-medium", description="Model to train")
    epochs: int = Field(default=3, description="Number of training epochs", ge=1, le=10)
    batch_size: int = Field(default=8, description="Training batch size", ge=1, le=32)
    learning_rate: float = Field(default=5e-5, description="Learning rate", ge=1e-6, le=1e-3)


class TrainingResponse(BaseModel):
    """Response schema for training results"""
    training_id: str = Field(..., description="Unique training identifier")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    model_path: Optional[str] = Field(default=None, description="Path to trained model")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Training metrics")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
