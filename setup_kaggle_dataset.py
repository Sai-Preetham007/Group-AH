#!/usr/bin/env python3
"""
Script to set up Kaggle disease dataset for training
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.kaggle_dataset_handler import KaggleDiseaseDatasetHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Set up Kaggle disease dataset"""
    try:
        logger.info("Setting up Kaggle disease dataset...")
        
        # Initialize dataset handler
        kaggle_handler = KaggleDiseaseDatasetHandler()
        logger.info("Dataset handler initialized")
        
        # Check if dataset files exist
        kaggle_dir = Path("data/raw/kaggle_disease_dataset")
        expected_files = ["symptom_precaution.csv", "dataset.csv", "symptom_Description.csv"]
        
        missing_files = []
        for file in expected_files:
            if not (kaggle_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing dataset files: {missing_files}")
            logger.info("Please download the dataset manually:")
            logger.info("1. Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
            logger.info("2. Download the dataset")
            logger.info(f"3. Extract files to: {kaggle_dir}")
            logger.info("4. Ensure these files are present:")
            for file in expected_files:
                logger.info(f"   - {file}")
            return
        
        # Load and process dataset
        logger.info("Loading dataset files...")
        datasets = kaggle_handler.load_disease_dataset()
        
        logger.info("Preprocessing dataset...")
        processed_df = kaggle_handler.preprocess_disease_data(datasets)
        
        logger.info("Creating Q&A pairs...")
        qa_df = kaggle_handler.create_medical_qa_pairs(processed_df)
        
        # Create train-test split
        logger.info("Creating train-test split...")
        train_df, test_df = kaggle_handler.create_train_test_split(qa_df, test_size=0.2)
        
        # Save processed data
        logger.info("Saving processed data...")
        kaggle_handler.save_processed_data(qa_df, "kaggle_disease_qa_full.csv")
        kaggle_handler.save_processed_data(train_df, "kaggle_disease_qa_train.csv")
        kaggle_handler.save_processed_data(test_df, "kaggle_disease_qa_test.csv")
        
        # Get statistics
        stats = kaggle_handler.get_dataset_statistics(qa_df)
        logger.info(f"Dataset statistics: {stats}")
        
        # Create knowledge base
        logger.info("Creating disease knowledge base...")
        knowledge_base = kaggle_handler.create_disease_knowledge_base(qa_df)
        
        # Save knowledge base
        import json
        with open("data/processed/disease_knowledge_base.json", "w") as f:
            json.dump(knowledge_base, f, indent=2)
        
        # Save statistics
        with open("data/processed/kaggle_dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset setup completed successfully!")
        logger.info(f"Total Q&A pairs: {len(qa_df)}")
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Diseases covered: {qa_df['disease'].nunique()}")
        
    except Exception as e:
        logger.error(f"Error during dataset setup: {e}")
        raise


if __name__ == "__main__":
    main()
