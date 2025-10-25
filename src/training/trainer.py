"""
Training module for Medical Knowledge RAG Chatbot
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb
from config import settings

logger = logging.getLogger(__name__)


class MedicalRAGTrainer:
    """Trainer for Medical RAG system"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.training_args = None
        
    def initialize_model(self, model_type: str = "causal_lm"):
        """Initialize model and tokenizer"""
        try:
            logger.info(f"Initializing {model_type} model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            elif model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model.to(self.device)
            logger.info(f"Model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def setup_training_args(self, output_dir: str = "data/models", 
                           num_epochs: int = 3, batch_size: int = 8,
                           learning_rate: float = 5e-5, warmup_steps: int = 100):
        """Setup training arguments"""
        try:
            self.training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if settings.wandb_project else None,
                run_name=f"medical_rag_{self.model_name.split('/')[-1]}"
            )
            
            logger.info("Training arguments configured")
            
        except Exception as e:
            logger.error(f"Error setting up training arguments: {e}")
            raise
    
    def train_model(self, train_dataset: DataLoader, eval_dataset: DataLoader = None,
                   custom_trainer: bool = False) -> Dict[str, Any]:
        """Train the model"""
        try:
            if custom_trainer:
                return self._custom_training_loop(train_dataset, eval_dataset)
            else:
                return self._huggingface_training(train_dataset, eval_dataset)
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _huggingface_training(self, train_dataset: DataLoader, 
                             eval_dataset: DataLoader = None) -> Dict[str, Any]:
        """Use HuggingFace Trainer for training"""
        try:
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            logger.info("Starting training with HuggingFace Trainer")
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.training_args.output_dir)
            
            # Evaluate
            eval_results = trainer.evaluate() if eval_dataset else {}
            
            return {
                "training_completed": True,
                "eval_results": eval_results,
                "model_path": self.training_args.output_dir
            }
            
        except Exception as e:
            logger.error(f"Error in HuggingFace training: {e}")
            raise
    
    def _custom_training_loop(self, train_dataset: DataLoader, 
                            eval_dataset: DataLoader = None) -> Dict[str, Any]:
        """Custom training loop with more control"""
        try:
            # Setup optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=self.training_args.learning_rate)
            
            total_steps = len(train_dataset) * self.training_args.num_train_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop
            self.model.train()
            global_step = 0
            training_loss = 0.0
            
            for epoch in range(self.training_args.num_train_epochs):
                epoch_loss = 0.0
                progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch+1}")
                
                for batch in progress_bar:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    training_loss += loss.item()
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({"loss": loss.item()})
                    
                    # Logging
                    if global_step % self.training_args.logging_steps == 0:
                        avg_loss = training_loss / self.training_args.logging_steps
                        logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                        training_loss = 0.0
                
                # Save checkpoint
                if (epoch + 1) % 1 == 0:  # Save every epoch
                    checkpoint_dir = Path(self.training_args.output_dir) / f"checkpoint-epoch-{epoch+1}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Final evaluation
            eval_results = self._evaluate_model(eval_dataset) if eval_dataset else {}
            
            # Save final model
            final_model_dir = Path(self.training_args.output_dir) / "final_model"
            final_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            
            return {
                "training_completed": True,
                "eval_results": eval_results,
                "model_path": str(final_model_dir),
                "total_steps": global_step
            }
            
        except Exception as e:
            logger.error(f"Error in custom training loop: {e}")
            raise
    
    def _evaluate_model(self, eval_dataset: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        try:
            self.model.eval()
            total_loss = 0.0
            predictions = []
            labels = []
            
            with torch.no_grad():
                for batch in tqdm(eval_dataset, desc="Evaluating"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                    
                    # Generate predictions
                    generated = self.model.generate(
                        batch['input_ids'],
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    predictions.extend(generated.cpu().numpy())
                    labels.extend(batch['labels'].cpu().numpy())
            
            # Calculate metrics
            avg_loss = total_loss / len(eval_dataset)
            
            return {
                "eval_loss": avg_loss,
                "num_samples": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def generate_response(self, input_text: str, max_length: int = 512) -> str:
        """Generate response using trained model"""
        try:
            self.model.eval()
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input text from response
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"
    
    def save_training_config(self, config: Dict[str, Any]) -> None:
        """Save training configuration"""
        try:
            config_path = Path(self.training_args.output_dir) / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Training config saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving training config: {e}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
