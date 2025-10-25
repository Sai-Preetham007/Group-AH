#!/usr/bin/env python3
"""
Command-line script to train medical language model
"""
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.kaggle_dataset_handler import KaggleDiseaseDatasetHandler
from training.trainer import MedicalRAGTrainer
from training.dataset_handler import create_data_loader
import torch
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train medical language model")
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
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Test set size (0.0 to 1.0)")
    parser.add_argument("--use_gpu", action="store_true", 
                       help="Use GPU if available")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting medical model training...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Check if dataset exists
        if not Path("data/processed/kaggle_disease_qa_train.csv").exists():
            logger.error("Training dataset not found. Please run setup_kaggle_dataset.py first.")
            return
        
        # Initialize dataset handler
        kaggle_handler = KaggleDiseaseDatasetHandler()
        
        # Load datasets
        logger.info("Loading datasets...")
        train_df = kaggle_handler.load_processed_data("kaggle_disease_qa_train.csv")
        test_df = kaggle_handler.load_processed_data("kaggle_disease_qa_test.csv")
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Initialize trainer
        device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        trainer = MedicalRAGTrainer(model_name=args.model_name, device=device)
        
        logger.info(f"Initializing model: {args.model_name}")
        trainer.initialize_model(model_type="causal_lm")
        
        # Create data loaders
        logger.info("Creating data loaders...")
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
        
        # Setup training arguments
        logger.info("Setting up training configuration...")
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
        
        # Start training
        logger.info("Starting training...")
        training_results = trainer.train_model(
            train_dataset=train_loader,
            eval_dataset=test_loader,
            custom_trainer=True
        )
        
        # Save results
        results_path = Path(args.output_dir) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Results saved to: {results_path}")
        
        # Test the model
        logger.info("Testing trained model...")
        test_questions = [
            "What are the symptoms of diabetes?",
            "How can I prevent heart disease?",
            "What is hypertension?"
        ]
        
        for question in test_questions:
            try:
                response = trainer.generate_response(question, max_length=200)
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")
            except Exception as e:
                logger.error(f"Error generating response for '{question}': {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
