#!/usr/bin/env python3
"""
Script to set up initial data for the Medical Knowledge RAG Chatbot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.medical_data_loader import MedicalDataLoader
from rag.vector_store import MedicalVectorStore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Set up initial data for the RAG system"""
    try:
        logger.info("Starting data setup...")
        
        # Initialize data loader
        data_loader = MedicalDataLoader()
        logger.info("Data loader initialized")
        
        # Load and process FDA drug data
        logger.info("Loading FDA drug data...")
        fda_drugs = data_loader.load_fda_drug_data(limit=500)
        processed_drugs = data_loader.preprocess_drug_data(fda_drugs)
        data_loader.save_processed_data(processed_drugs, "fda_drugs_processed.json")
        logger.info(f"Processed {len(processed_drugs)} drug records")
        
        # Load and process clinical trials
        logger.info("Loading clinical trials data...")
        trials = data_loader.load_clinical_trials(condition="diabetes", limit=200)
        processed_trials = data_loader.preprocess_clinical_trials(trials)
        data_loader.save_processed_data(processed_trials, "clinical_trials_processed.json")
        logger.info(f"Processed {len(processed_trials)} trial records")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = MedicalVectorStore()
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        vector_store.add_documents(processed_drugs)
        vector_store.add_documents(processed_trials)
        
        # Get statistics
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store setup complete. Total documents: {stats.get('total_documents', 0)}")
        
        logger.info("Data setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data setup: {e}")
        raise


if __name__ == "__main__":
    main()
