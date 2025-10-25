#!/usr/bin/env python3
"""
Model Training Script

This script trains the Medical Knowledge RAG Chatbot models.
"""

import sys
import os
sys.path.append('../')

import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset, DataLoader

class MedicalQADataset(Dataset):
    """PyTorch Dataset for medical Q&A data"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Format for causal language modeling
        text = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def load_training_data():
    """Load training data"""
    data_path = Path("../../data/processed/train_qa_pairs.json")
    if data_path.exists():
        with open(data_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training samples")
        return data
    else:
        print("Training data not found. Please run data processing first.")
        return []

def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    model_name = "microsoft/DialoGPT-medium"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model loaded: {model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

def train_model(model, tokenizer, train_data):
    """Train the model"""
    # Create dataset
    train_dataset = MedicalQADataset(train_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="../../data/models/medical_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="../../logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained("../../data/models/medical_model")
    
    print("Training completed!")
    print("Model saved to ../../data/models/medical_model/")

def main():
    """Main training function"""
    print("Starting model training...")
    
    # Load training data
    train_data = load_training_data()
    if not train_data:
        return
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Train model
    train_model(model, tokenizer, train_data)
    
    print("Model training completed!")

if __name__ == "__main__":
    main()
