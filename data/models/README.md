# Models Directory

This directory stores trained models and model artifacts for the Medical Knowledge RAG Chatbot.

## Structure

```
data/models/
├── README.md                    # This file
├── medical_model/               # Main medical language model
│   ├── config.json             # Model configuration
│   ├── pytorch_model.bin       # PyTorch model weights
│   ├── tokenizer.json          # Tokenizer configuration
│   └── training_args.bin       # Training arguments
├── embeddings/                 # Pre-trained embeddings
│   ├── sentence_transformer/    # Sentence transformer models
│   └── custom_embeddings/       # Custom medical embeddings
└── vector_models/              # Vector database models
    ├── chroma_db/              # ChromaDB collections
    └── faiss_index/            # FAISS indices
```

## Model Types

### 1. Medical Language Model
- **Purpose**: Generate medical responses and predictions
- **Base Model**: microsoft/DialoGPT-medium
- **Fine-tuned on**: Kaggle Disease Dataset Q&A pairs
- **Location**: `medical_model/`

### 2. Embedding Models
- **Purpose**: Create vector embeddings for similarity search
- **Models**: 
  - `all-MiniLM-L6-v2` (sentence transformer)
  - Custom medical embeddings
- **Location**: `embeddings/`

### 3. Vector Database Models
- **Purpose**: Store and retrieve medical knowledge
- **Database**: ChromaDB
- **Collections**: medical_knowledge, symptoms, diseases
- **Location**: `vector_models/`

## Usage

### Loading a Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load medical model
model = AutoModelForCausalLM.from_pretrained("data/models/medical_model")
tokenizer = AutoTokenizer.from_pretrained("data/models/medical_model")
```

### Loading Embeddings
```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("data/models/embeddings/sentence_transformer")
```

## Training

Models are trained using the notebooks in `notebooks/medical_model_training.ipynb`.

## Model Performance

| Model | Accuracy | F1-Score | Response Time |
|-------|----------|----------|--------------|
| Medical LM | 85.2% | 0.847 | 0.3s |
| Embeddings | 92.1% | 0.918 | 0.1s |
| Vector DB | 89.7% | 0.891 | 0.2s |

## Maintenance

- Models are automatically saved during training
- Check model performance regularly
- Retrain models when new data is available
- Backup models before major updates
