"""
API routes for Medical Knowledge RAG Chatbot
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
import logging
import time
import uuid
from datetime import datetime

from .schemas import (
    QueryRequest, QueryResponse, HealthResponse, StatsResponse,
    ValidationRequest, ValidationResponse, TrainingRequest, TrainingResponse
)
from ..rag.rag_engine import MedicalRAGEngine
from ..evaluation.medical_guardrails import MedicalGuardrails
from ..utils.medical_validator import MedicalValidator
from ..utils.helpers import validate_medical_query, format_medical_response

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global variables (would be injected via dependency injection in production)
rag_engine: MedicalRAGEngine = None
guardrails: MedicalGuardrails = None
validator: MedicalValidator = None


def get_rag_engine() -> MedicalRAGEngine:
    """Dependency to get RAG engine"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag_engine


def get_guardrails() -> MedicalGuardrails:
    """Dependency to get guardrails"""
    if guardrails is None:
        raise HTTPException(status_code=503, detail="Guardrails not initialized")
    return guardrails


def get_validator() -> MedicalValidator:
    """Dependency to get validator"""
    if validator is None:
        raise HTTPException(status_code=503, detail="Validator not initialized")
    return validator


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    try:
        if rag_engine:
            stats = rag_engine.vector_store.get_collection_stats()
            model_info = {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_model": "gpt-3.5-turbo"
            }
        else:
            stats = None
            model_info = None
        
        return HealthResponse(
            status="healthy",
            message="Medical Knowledge RAG Chatbot is running",
            vector_store_stats=stats,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="error",
            message=f"Error: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


@router.post("/query", response_model=QueryResponse)
async def query_medical_knowledge(
    request: QueryRequest,
    rag_engine: MedicalRAGEngine = Depends(get_rag_engine),
    guardrails: MedicalGuardrails = Depends(get_guardrails)
):
    """Query the medical knowledge base"""
    try:
        start_time = time.time()
        
        # Validate query
        query_validation = validate_medical_query(request.question)
        if not query_validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid query: {query_validation['warnings']}"
            )
        
        # Process query
        logger.info(f"Processing query: {request.question}")
        result = rag_engine.query(request.question)
        
        # Apply safety guardrails
        if request.safety_check:
            safety_result = guardrails.validate_response(
                result["response"], 
                result["sources"]
            )
            
            if not safety_result["is_safe"]:
                logger.warning(f"Unsafe response detected: {safety_result['warnings']}")
                # Add safety warnings to response
                result["response"] = guardrails.add_safety_disclaimers(result["response"])
        
        # Format response
        if request.include_sources:
            result["response"] = format_medical_response(
                result["response"], 
                result["sources"]
            )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=result["response"],
            sources=result["sources"],
            context_used=result["context_used"],
            query=result["query"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(rag_engine: MedicalRAGEngine = Depends(get_rag_engine)):
    """Get system statistics"""
    try:
        stats = rag_engine.vector_store.get_collection_stats()
        model_info = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "gpt-3.5-turbo"
        }
        system_info = {
            "version": "1.0.0",
            "uptime": time.time(),  # Would be actual uptime in production
            "status": "running"
        }
        
        return StatsResponse(
            vector_store_stats=stats,
            model_info=model_info,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_content(
    request: ValidationRequest,
    validator: MedicalValidator = Depends(get_validator)
):
    """Validate medical content for safety"""
    try:
        if request.validation_type == "medical":
            result = validator.validate_query(request.text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported validation type")
        
        return ValidationResponse(
            is_safe=result.is_safe,
            confidence=result.confidence,
            warnings=result.warnings,
            suggestions=result.suggestions,
            risk_level=result.risk_level
        )
    except Exception as e:
        logger.error(f"Error validating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_vector_store(rag_engine: MedicalRAGEngine = Depends(get_rag_engine)):
    """Reset the vector store (admin endpoint)"""
    try:
        rag_engine.vector_store.reset_collection()
        return {"message": "Vector store reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start model training (background task)"""
    try:
        training_id = str(uuid.uuid4())
        
        # Add training task to background
        background_tasks.add_task(
            _run_training,
            training_id,
            request.dataset_path,
            request.model_name,
            request.epochs,
            request.batch_size,
            request.learning_rate
        )
        
        return TrainingResponse(
            training_id=training_id,
            status="started",
            message="Training started in background"
        )
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_training(training_id: str, dataset_path: str, model_name: str, 
                      epochs: int, batch_size: int, learning_rate: float):
    """Background training task"""
    try:
        logger.info(f"Starting training {training_id}")
        # Training implementation would go here
        # This is a placeholder
        logger.info(f"Training {training_id} completed")
    except Exception as e:
        logger.error(f"Training {training_id} failed: {e}")


# Initialize global components
def initialize_components(rag_engine_instance: MedicalRAGEngine, 
                         guardrails_instance: MedicalGuardrails,
                         validator_instance: MedicalValidator):
    """Initialize global components"""
    global rag_engine, guardrails, validator
    rag_engine = rag_engine_instance
    guardrails = guardrails_instance
    validator = validator_instance