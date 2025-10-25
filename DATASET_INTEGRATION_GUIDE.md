# Dataset Integration Guide

## Kaggle Disease Dataset Integration

This guide explains how to integrate the Kaggle disease-symptom dataset into your Medical Knowledge RAG Chatbot.

### Dataset Information
- **Source**: [Kaggle Disease Symptom Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- **Files**: 
  - `symptom_precaution.csv` - Disease precautions
  - `dataset.csv` - Main disease-symptom data
  - `symptom_Description.csv` - Disease descriptions
- **Content**: Disease symptoms, precautions, and descriptions for medical Q&A training

### Step 1: Download Dataset

#### Option A: Manual Download (Recommended)
1. Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
2. Click "Download" button
3. Extract the zip file to: `data/raw/kaggle_disease_dataset/`
4. Ensure these files are present:
   - `symptom_precaution.csv`
   - `dataset.csv`
   - `symptom_Description.csv`

#### Option B: Kaggle API (Advanced)
```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials
# Place kaggle.json in ~/.kaggle/ directory

# Download dataset
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p data/raw/kaggle_disease_dataset/
```

### Step 2: Process Dataset

Run the dataset processing script:

```bash
python setup_kaggle_dataset.py
```

This will:
- Load all dataset files
- Preprocess and clean the data
- Create Q&A pairs for training
- Split into train/test sets
- Save processed data

### Step 3: Training Options

#### Option A: Use Jupyter Notebooks
1. **Dataset Integration**: `notebooks/kaggle_dataset_integration.ipynb`
   - Download and process dataset
   - Create Q&A pairs
   - Visualize data distribution
   - Prepare for training

2. **Model Training**: `notebooks/medical_model_training.ipynb`
   - Train medical language model
   - Fine-tune on disease data
   - Evaluate performance
   - Save trained model

#### Option B: Command Line Training
```bash
# Process dataset
python setup_kaggle_dataset.py

# Train model (if you have GPU)
python -m src.training.train_medical_model.py
```

### Step 4: Dataset Structure

After processing, you'll have:

```
data/
├── raw/
│   └── kaggle_disease_dataset/
│       ├── symptom_precaution.csv
│       ├── dataset.csv
│       └── symptom_Description.csv
├── processed/
│   ├── kaggle_disease_qa_full.csv      # All Q&A pairs
│   ├── kaggle_disease_qa_train.csv     # Training set
│   ├── kaggle_disease_qa_test.csv      # Test set
│   ├── kaggle_rag_data.json           # RAG-ready format
│   ├── disease_knowledge_base.json    # Structured knowledge
│   └── kaggle_dataset_stats.json      # Dataset statistics
└── models/
    └── medical_dialogue_model/         # Trained model
```

### Step 5: Dataset Statistics

The processed dataset typically contains:
- **Total Q&A pairs**: ~10,000+ (varies by dataset)
- **Diseases covered**: 100+ medical conditions
- **Categories**: Symptoms, Precautions, Descriptions
- **Text length**: Average 200-500 characters per Q&A pair

### Step 6: Integration with RAG System

The processed data is automatically formatted for RAG integration:

```python
# Load RAG-ready data
import json
with open('data/processed/kaggle_rag_data.json', 'r') as f:
    rag_data = json.load(f)

# Add to vector store
from src.rag.vector_store import MedicalVectorStore
vector_store = MedicalVectorStore()
vector_store.add_documents(rag_data)
```

### Step 7: Training Configuration

Key training parameters:

```python
training_config = {
    "model_name": "microsoft/DialoGPT-medium",
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

### Step 8: Evaluation Metrics

The training includes:
- **Loss metrics**: Training and validation loss
- **Generation quality**: Response relevance and coherence
- **Medical accuracy**: Disease-specific knowledge retention
- **Safety checks**: Medical safety guardrails

### Troubleshooting

#### Common Issues:

1. **Missing dataset files**
   - Ensure all CSV files are downloaded and extracted
   - Check file paths in `data/raw/kaggle_disease_dataset/`

2. **Memory issues during training**
   - Reduce batch size (e.g., from 8 to 4)
   - Use gradient accumulation
   - Enable mixed precision training

3. **CUDA out of memory**
   - Reduce max_length (e.g., from 512 to 256)
   - Use smaller model (e.g., DialoGPT-small)
   - Enable gradient checkpointing

4. **Poor training performance**
   - Increase number of epochs
   - Adjust learning rate
   - Check data quality and preprocessing

### Next Steps

After successful training:

1. **Test the model** with medical queries
2. **Integrate with RAG system** for enhanced responses
3. **Deploy the model** for production use
4. **Monitor performance** and retrain as needed

### Support

For issues or questions:
- Check the Jupyter notebooks for detailed examples
- Review the training logs for error messages
- Ensure all dependencies are installed correctly
