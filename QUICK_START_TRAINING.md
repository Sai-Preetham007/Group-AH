# Quick Start Training Guide

## 🚀 Get Started with Medical Model Training

This guide will help you quickly set up and train your medical language model using the Kaggle disease dataset.

## Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download Dataset**
   - Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   - Download and extract to: `data/raw/kaggle_disease_dataset/`
   - Ensure these files exist:
     - `symptom_precaution.csv`
     - `dataset.csv`
     - `symptom_Description.csv`

## Quick Start Options

### Option 1: Jupyter Notebooks (Recommended for Beginners)

1. **Dataset Integration**
   ```bash
   jupyter notebook notebooks/kaggle_dataset_integration.ipynb
   ```
   - Run all cells to process the dataset
   - Visualize data distribution
   - Create Q&A pairs

2. **Model Training**
   ```bash
   jupyter notebook notebooks/medical_model_training.ipynb
   ```
   - Train the medical language model
   - Evaluate performance
   - Test with sample queries

### Option 2: Command Line (Advanced Users)

1. **Process Dataset**
   ```bash
   python setup_kaggle_dataset.py
   ```

2. **Train Model**
   ```bash
   # Basic training
   python src/training/train_medical_model.py
   
   # Advanced training with custom parameters
   python src/training/train_medical_model.py \
     --model_name microsoft/DialoGPT-medium \
     --epochs 5 \
     --batch_size 4 \
     --learning_rate 3e-5 \
     --use_gpu
   ```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `microsoft/DialoGPT-medium` | Pre-trained model to use |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `8` | Batch size (reduce if memory issues) |
| `--learning_rate` | `5e-5` | Learning rate for training |
| `--max_length` | `512` | Maximum sequence length |
| `--output_dir` | `data/models/medical_dialogue_model` | Output directory |
| `--use_gpu` | `False` | Use GPU if available |

## Expected Results

After successful training, you'll have:

- **Trained Model**: `data/models/medical_dialogue_model/`
- **Training Results**: `data/models/training_results.json`
- **Processed Data**: `data/processed/kaggle_disease_qa_*.csv`

## Testing Your Model

```python
from src.training.trainer import MedicalRAGTrainer

# Load trained model
trainer = MedicalRAGTrainer()
trainer.load_model("data/models/medical_dialogue_model/final_model")

# Test with medical queries
questions = [
    "What are the symptoms of diabetes?",
    "How can I prevent heart disease?",
    "What is hypertension?"
]

for question in questions:
    response = trainer.generate_response(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

## Troubleshooting

### Common Issues:

1. **"Dataset not found" error**
   - Ensure you've downloaded the Kaggle dataset
   - Check file paths in `data/raw/kaggle_disease_dataset/`

2. **CUDA out of memory**
   - Reduce batch size: `--batch_size 4`
   - Reduce max length: `--max_length 256`
   - Use CPU: Remove `--use_gpu` flag

3. **Poor training performance**
   - Increase epochs: `--epochs 5`
   - Adjust learning rate: `--learning_rate 3e-5`
   - Check data quality

4. **Slow training**
   - Use GPU: Add `--use_gpu` flag
   - Reduce batch size if memory limited
   - Use smaller model: `--model_name microsoft/DialoGPT-small`

## Next Steps

After training:

1. **Integrate with RAG System**
   ```python
   # Load trained model for RAG
   from src.training.trainer import MedicalRAGTrainer
   trainer = MedicalRAGTrainer()
   trainer.load_model("data/models/medical_dialogue_model/final_model")
   ```

2. **Deploy Model**
   - Use the trained model in your RAG system
   - Test with real medical queries
   - Monitor performance

3. **Improve Model**
   - Collect more medical data
   - Fine-tune on specific diseases
   - Implement advanced training techniques

## Support

- **Documentation**: Check `DATASET_INTEGRATION_GUIDE.md`
- **Notebooks**: Use Jupyter notebooks for step-by-step guidance
- **Logs**: Check training logs for error messages
- **Issues**: Review common troubleshooting solutions above

## Performance Tips

- **GPU Training**: Use `--use_gpu` for faster training
- **Memory Optimization**: Reduce batch size if needed
- **Data Quality**: Ensure clean, well-formatted data
- **Model Selection**: Choose appropriate model size for your hardware

Happy training! 🎉
