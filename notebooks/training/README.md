# Training Notebooks

This directory contains notebooks for training and fine-tuning the Medical Knowledge RAG Chatbot models.

## Notebooks

### 1. `basic_training.ipynb`
Basic model training notebook:
- Simple training setup
- Basic hyperparameter tuning
- Training progress monitoring
- Model saving and loading

### 2. `advanced_training.ipynb`
Advanced training techniques:
- Transfer learning
- Multi-task learning
- Ensemble methods
- Hyperparameter optimization

### 3. `fine_tuning.ipynb`
Fine-tuning pre-trained models:
- Domain-specific fine-tuning
- Medical terminology adaptation
- Safety constraint training
- Performance optimization

### 4. `hyperparameter_tuning.ipynb`
Hyperparameter optimization:
- Grid search
- Random search
- Bayesian optimization
- Automated hyperparameter tuning

## Training Approaches

### 1. Basic Training
- **Purpose**: Initial model training
- **Dataset**: Kaggle Disease Dataset
- **Model**: DialoGPT-medium
- **Technique**: Standard fine-tuning

### 2. Advanced Training
- **Purpose**: Improved performance
- **Techniques**: Transfer learning, multi-task learning
- **Optimization**: Advanced optimization algorithms
- **Regularization**: Dropout, weight decay

### 3. Fine-tuning
- **Purpose**: Domain-specific adaptation
- **Base Model**: Pre-trained medical models
- **Technique**: Gradual unfreezing
- **Focus**: Medical terminology and safety

### 4. Hyperparameter Tuning
- **Purpose**: Optimal parameter selection
- **Methods**: Grid search, random search, Bayesian
- **Metrics**: Accuracy, F1-score, response time
- **Tools**: Optuna, Weights & Biases

## Training Configuration

### Model Parameters
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "max_length": 512,
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "warmup_steps": 100,
  "weight_decay": 0.01
}
```

### Training Setup
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./models/medical_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

## Training Pipeline

### 1. Data Preparation
- Load and preprocess dataset
- Create train/validation/test splits
- Tokenize data
- Create data loaders

### 2. Model Setup
- Load pre-trained model
- Configure tokenizer
- Set up training arguments
- Initialize trainer

### 3. Training Execution
- Start training process
- Monitor training progress
- Save checkpoints
- Evaluate on validation set

### 4. Model Evaluation
- Test on test set
- Calculate performance metrics
- Generate evaluation report
- Save final model

## Monitoring Training

### Training Metrics
- **Loss**: Training and validation loss
- **Accuracy**: Prediction accuracy
- **F1-Score**: F1 score for classification
- **BLEU Score**: Text generation quality

### Visualization
- **Loss Curves**: Training and validation loss over time
- **Accuracy Curves**: Accuracy improvement over epochs
- **Learning Rate**: Learning rate schedule
- **Gradient Norms**: Gradient magnitude monitoring

### Logging
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking
- **MLflow**: Model versioning and tracking
- **Custom Logging**: Application-specific metrics

## Best Practices

### 1. Data Quality
- **Clean Data**: Remove noisy samples
- **Balanced Dataset**: Ensure class balance
- **Data Augmentation**: Increase dataset diversity
- **Validation Split**: Proper train/validation split

### 2. Model Training
- **Learning Rate**: Start with small learning rate
- **Batch Size**: Use appropriate batch size
- **Regularization**: Apply dropout and weight decay
- **Early Stopping**: Prevent overfitting

### 3. Monitoring
- **Track Metrics**: Monitor key performance indicators
- **Visualize Progress**: Use plotting libraries
- **Save Checkpoints**: Regular model saving
- **Experiment Tracking**: Log all experiments

### 4. Evaluation
- **Test Set**: Use held-out test set
- **Cross-Validation**: Validate model robustness
- **A/B Testing**: Compare different models
- **User Feedback**: Collect real-world feedback

## Troubleshooting

### Common Issues
- **Overfitting**: Reduce model complexity or increase regularization
- **Underfitting**: Increase model capacity or training time
- **Slow Training**: Optimize batch size and learning rate
- **Memory Issues**: Reduce batch size or use gradient accumulation

### Solutions
- **Hyperparameter Tuning**: Optimize model parameters
- **Data Augmentation**: Increase dataset size
- **Model Architecture**: Experiment with different architectures
- **Training Strategy**: Try different training approaches
