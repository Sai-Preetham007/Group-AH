#!/usr/bin/env python3
"""
Complete training pipeline for Medical Knowledge RAG Chatbot
"""
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.kaggle_dataset_handler import KaggleDiseaseDatasetHandler
from src.training.trainer import MedicalRAGTrainer
from src.training.dataset_handler import create_data_loader
import torch
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Complete training pipeline"""
    parser = argparse.ArgumentParser(description="Complete Medical RAG Training Pipeline")
    parser.add_argument("--model_name", default="microsoft/DialoGPT-medium", 
                       help="Model name to use for training")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", default="data/models/medical_dialogue_model", 
                       help="Output directory for trained model")
    parser.add_argument("--use_gpu", action="store_true", 
                       help="Use GPU if available")
    parser.add_argument("--skip_data_setup", action="store_true", 
                       help="Skip dataset processing if already done")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting complete Medical RAG training pipeline...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Step 1: Setup dataset (if not skipped)
        if not args.skip_data_setup:
            logger.info("Step 1: Setting up Kaggle dataset...")
            kaggle_handler = KaggleDiseaseDatasetHandler()
            
            # Check if dataset exists
            if not Path("data/processed/kaggle_disease_qa_train.csv").exists():
                logger.info("Processing Kaggle dataset...")
                datasets = kaggle_handler.load_disease_dataset()
                processed_df = kaggle_handler.preprocess_disease_data(datasets)
                qa_df = kaggle_handler.create_medical_qa_pairs(processed_df)
                train_df, test_df = kaggle_handler.create_train_test_split(qa_df, test_size=0.2)
                
                # Save processed data
                kaggle_handler.save_processed_data(qa_df, "kaggle_disease_qa_full.csv")
                kaggle_handler.save_processed_data(train_df, "kaggle_disease_qa_train.csv")
                kaggle_handler.save_processed_data(test_df, "kaggle_disease_qa_test.csv")
                
                logger.info(f"Dataset processed: {len(qa_df)} total, {len(train_df)} train, {len(test_df)} test")
            else:
                logger.info("Dataset already processed, skipping...")
        
        # Step 2: Load datasets
        logger.info("Step 2: Loading processed datasets...")
        kaggle_handler = KaggleDiseaseDatasetHandler()
        train_df = kaggle_handler.load_processed_data("kaggle_disease_qa_train.csv")
        test_df = kaggle_handler.load_processed_data("kaggle_disease_qa_test.csv")
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Step 3: Initialize trainer
        logger.info("Step 3: Initializing trainer...")
        device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        trainer = MedicalRAGTrainer(model_name=args.model_name, device=device)
        
        logger.info(f"Initializing model: {args.model_name}")
        trainer.initialize_model(model_type="causal_lm")
        
        # Step 4: Create data loaders
        logger.info("Step 4: Creating data loaders...")
        train_loader = create_data_loader(
            train_df, trainer.tokenizer, 
            batch_size=args.batch_size, 
            max_length=args.max_length, 
            shuffle=True
        )
        
        test_loader = create_data_loader(
            test_df, trainer.tokenizer, 
            batch_size=args.batch_size, 
            max_length=args.max_length, 
            shuffle=False
        )
        
        # Step 5: Setup training
        logger.info("Step 5: Setting up training configuration...")
        trainer.setup_training_args(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=100
        )
        
        # Save training configuration
        training_config = {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "device": device,
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        }
        
        trainer.save_training_config(training_config)
        
        # Step 6: Train model
        logger.info("Step 6: Starting model training...")
        training_results = trainer.train_model(
            train_dataset=train_loader,
            eval_dataset=test_loader,
            custom_trainer=True
        )
        
        # Step 7: Save results
        logger.info("Step 7: Saving training results...")
        results_path = Path(args.output_dir) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Step 8: Test model
        logger.info("Step 8: Testing trained model...")
        test_questions = [
            "What are the symptoms of diabetes?",
            "How can I prevent heart disease?",
            "What is hypertension?",
            "What precautions should I take for asthma?"
        ]
        
        logger.info("Testing model with sample questions:")
        for question in test_questions:
            try:
                response = trainer.generate_response(question, max_length=200)
                logger.info(f"Q: {question}")
                logger.info(f"A: {response[:100]}...")
            except Exception as e:
                logger.error(f"Error testing question '{question}': {e}")
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
