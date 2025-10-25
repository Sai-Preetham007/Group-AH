# Training Data Directory

This directory contains training-specific data, configurations, and artifacts for model training.

## Structure

```
data/training/
├── README.md                    # This file
├── configs/                    # Training configurations
│   ├── model_config.json       # Model hyperparameters
│   ├── training_config.json    # Training parameters
│   └── data_config.json        # Data processing config
├── datasets/                   # Training datasets
│   ├── train_dataset.json      # Training dataset
│   ├── val_dataset.json        # Validation dataset
│   └── test_dataset.json       # Test dataset
├── checkpoints/                # Training checkpoints
│   ├── epoch_1/                # Checkpoint after epoch 1
│   ├── epoch_2/                # Checkpoint after epoch 2
│   └── best_model/             # Best performing model
├── logs/                       # Training logs
│   ├── training.log            # Training progress
│   ├── metrics.json            # Training metrics
│   └── tensorboard/            # TensorBoard logs
└── experiments/                # Experiment tracking
    ├── exp_001/                # Experiment 1
    ├── exp_002/                # Experiment 2
    └── exp_003/                # Experiment 3
```

## Configuration Files

### Model Configuration (`configs/model_config.json`)
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "max_length": 512,
  "num_labels": 41,
  "hidden_size": 768,
  "num_attention_heads": 12,
  "num_hidden_layers": 12
}
```

### Training Configuration (`configs/training_config.json`)
```json
{
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 1,
  "max_grad_norm": 1.0
}
```

## Training Datasets

### Training Dataset (`datasets/train_dataset.json`)
Contains the training data with:
- Question-answer pairs
- Disease labels
- Symptom information
- Context data

### Validation Dataset (`datasets/val_dataset.json`)
Contains validation data for:
- Model evaluation during training
- Hyperparameter tuning
- Early stopping

## Checkpoints

Training checkpoints are saved at regular intervals:
- **epoch_1/**: After 1 epoch
- **epoch_2/**: After 2 epochs
- **best_model/**: Best performing model

## Training Logs

### Training Progress (`logs/training.log`)
```
2024-01-15 10:30:00 - Starting training
2024-01-15 10:30:15 - Epoch 1/3, Step 10/1000, Loss: 2.345
2024-01-15 10:30:30 - Epoch 1/3, Step 20/1000, Loss: 2.123
...
```

### Metrics (`logs/metrics.json`)
```json
{
  "epoch": 3,
  "train_loss": 1.234,
  "val_loss": 1.456,
  "accuracy": 0.852,
  "f1_score": 0.847,
  "bleu_score": 0.623
}
```

## Experiment Tracking

Each experiment is tracked with:
- **Configuration**: Model and training parameters
- **Results**: Performance metrics
- **Artifacts**: Trained models and logs
- **Notes**: Experiment description and findings

## Usage

### Starting Training
```bash
# Run training with default config
python run_training.py

# Run training with custom config
python run_training.py --config data/training/configs/custom_config.json
```

### Loading Checkpoints
```python
from transformers import AutoModelForCausalLM

# Load from checkpoint
model = AutoModelForCausalLM.from_pretrained("data/training/checkpoints/best_model")
```

### Viewing Training Progress
```bash
# View training logs
tail -f data/training/logs/training.log

# View TensorBoard
tensorboard --logdir data/training/logs/tensorboard
```

## Best Practices

1. **Save checkpoints regularly** to avoid losing progress
2. **Monitor training metrics** to detect overfitting
3. **Use validation data** for model selection
4. **Track experiments** to compare different approaches
5. **Clean up old checkpoints** to save disk space
