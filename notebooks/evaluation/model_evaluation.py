#!/usr/bin/env python3
"""
Model Evaluation Script

This script evaluates the performance of the Medical Knowledge RAG Chatbot models.
"""

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path

def load_test_data():
    """Load test data for evaluation"""
    test_data_path = Path("../../data/processed/test_qa_pairs.json")
    if test_data_path.exists():
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    else:
        print("Test data not found. Please run data processing first.")
        return []

def evaluate_model(test_data):
    """Evaluate model performance"""
    if not test_data:
        print("No test data available for evaluation")
        return None
    
    # Mock evaluation results (replace with actual model evaluation)
    accuracy = 0.852
    precision = 0.847
    recall = 0.856
    f1 = 0.851
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def visualize_results(metrics):
    """Create performance visualization"""
    if not metrics:
        print("No metrics available for visualization")
        return
    
    plt.figure(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics['accuracy'], metrics['precision'], 
              metrics['recall'], metrics['f1_score']]
    
    plt.bar(metric_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def save_results(metrics, test_data):
    """Save evaluation results"""
    if not metrics or not test_data:
        print("No results to save")
        return
    
    results_dir = Path("../../data/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluation_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "test_samples": len(test_data),
        "metrics": metrics
    }
    
    with open(results_dir / "model_evaluation.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation results saved to {results_dir / 'model_evaluation.json'}")

def main():
    """Main evaluation function"""
    print("Starting model evaluation...")
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate model
    metrics = evaluate_model(test_data)
    
    # Visualize results
    visualize_results(metrics)
    
    # Save results
    save_results(metrics, test_data)
    
    print("Model evaluation completed!")

if __name__ == "__main__":
    main()
