# Medical Knowledge RAG Chatbot (Healthcare Domain)

## Problem Statement

In the healthcare industry, accurate and trustworthy information retrieval is critical. Patients and healthcare practitioners often face difficulties finding reliable medical information across sources such as drug databases, clinical trial summaries, and treatment guidelines. Existing chatbots tend to hallucinate or provide unsafe, non-cited responses.

## Goal

Build a Retrieval-Augmented Generation (RAG) powered Medical Knowledge Chatbot that can accurately answer medical and pharmaceutical queries by retrieving facts from verified public datasets (e.g., openFDA, WHO guidelines, or clinical trial summaries) and providing source-cited, safe, and human-readable responses.

## Solution Overview

This project implements a comprehensive Medical Knowledge RAG Chatbot with the following key features:

- **Data Sources**: FDA drug database, ClinicalTrials.gov, WHO guidelines
- **RAG Architecture**: Vector-based retrieval with LLM generation
- **Safety Guardrails**: Medical safety checks and disclaimers
- **Evaluation Framework**: Comprehensive metrics for accuracy and safety
- **API Service**: FastAPI-based REST API for easy integration

## Project Structure

```
Group-AH/
├── src/                          # Source code
│   ├── data_processing/          # Data ingestion and preprocessing
│   │   └── medical_data_loader.py
│   ├── rag/                      # RAG system components
│   │   ├── vector_store.py       # Vector database operations
│   │   └── rag_engine.py         # RAG engine implementation
│   ├── api/                      # FastAPI service
│   │   └── main.py               # API endpoints
│   ├── evaluation/               # Evaluation and guardrails
│   │   ├── medical_guardrails.py # Safety guardrails
│   │   └── evaluator.py          # Evaluation framework
│   └── utils/                    # Utility functions
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   ├── chroma_db/                # Vector database
│   └── evaluation/                # Test cases and results
├── notebooks/                     # Jupyter notebooks
│   ├── data_ingestion.ipynb      # Data loading demo
│   └── rag_testing.ipynb         # RAG system testing
├── tests/                        # Test cases
│   └── test_rag_system.py        # Unit tests
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── run_server.py                 # Server startup script
├── setup_data.py                 # Data initialization script
└── README.md                     # This file
```

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
VECTOR_DB_PATH=./data/vector_db

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
```

### 3. Initialize Data

```bash
python setup_data.py
```

This will:
- Load FDA drug data
- Load clinical trials data
- Process and store in vector database

### 4. Run the API Server

```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

## Usage

### API Endpoints

- `GET /` - Health check
- `GET /health` - System health status
- `POST /query` - Query medical knowledge
- `GET /stats` - System statistics
- `POST /reset` - Reset vector store (admin)

### Example API Usage

```python
import requests

# Query the medical knowledge base
response = requests.post("http://localhost:8000/query", json={
    "question": "What are the side effects of metformin?"
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Sources: {result['sources']}")
```

### Jupyter Notebooks

- `notebooks/data_ingestion.ipynb` - Data loading and preprocessing
- `notebooks/rag_testing.ipynb` - RAG system testing and evaluation

## Key Features

### 1. Medical Data Sources
- **FDA Drug Database**: Drug labels, adverse events, recalls
- **Clinical Trials**: Research studies and trial data
- **WHO Guidelines**: International health recommendations

### 2. RAG System
- **Vector Store**: ChromaDB for efficient similarity search
- **Embeddings**: Sentence transformers for semantic understanding
- **LLM Integration**: OpenAI GPT for response generation

### 3. Safety Guardrails
- **Medical Safety Checks**: Detect unsafe medical advice
- **Emergency Detection**: Identify urgent medical situations
- **Source Validation**: Ensure credible medical sources
- **Disclaimers**: Automatic medical disclaimers

### 4. Evaluation Framework
- **ROUGE Scores**: Text similarity metrics
- **BLEU Scores**: Translation quality metrics
- **Faithfulness**: Source alignment evaluation
- **Safety Metrics**: Medical safety assessment

## Testing

Run the test suite:

```bash
pytest tests/test_rag_system.py -v
```

## Evaluation

The system includes comprehensive evaluation metrics:

- **Accuracy**: ROUGE, BLEU scores
- **Safety**: Medical guardrails validation
- **Faithfulness**: Source-based verification
- **Source Quality**: Credibility assessment

## Limitations and Assumptions

1. **Data Sources**: Limited to publicly available medical databases
2. **Medical Advice**: System provides information, not medical advice
3. **Accuracy**: Responses depend on source data quality
4. **Real-time**: Data may not reflect latest medical developments

## ML Training with Kaggle Dataset

### Dataset Integration
- **Kaggle Dataset**: Disease-symptom dataset with 100+ medical conditions
- **Data Processing**: Automated Q&A pair generation
- **Training Pipeline**: Complete ML training workflow
- **Model Evaluation**: Comprehensive performance metrics

### Quick Start Training

1. **Download Dataset**:
   ```bash
   # Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   # Download and extract to: data/raw/kaggle_disease_dataset/
   ```

2. **Process Dataset**:
   ```bash
   python setup_kaggle_dataset.py
   ```

3. **Train Model**:
   ```bash
   # Option A: Jupyter Notebooks (Recommended)
   jupyter notebook notebooks/kaggle_dataset_integration.ipynb
   jupyter notebook notebooks/medical_model_training.ipynb
   
   # Option B: Command Line
   python src/training/train_medical_model.py --use_gpu
   ```

### Training Features
- **Medical Q&A Pairs**: 10,000+ disease-related questions and answers
- **Disease Categories**: Symptoms, precautions, descriptions
- **Model Fine-tuning**: DialoGPT-based medical dialogue model
- **Evaluation Metrics**: ROUGE, BLEU, medical accuracy
- **Safety Guardrails**: Medical safety validation

### Training Documentation
- `DATASET_INTEGRATION_GUIDE.md` - Complete dataset integration guide
- `QUICK_START_TRAINING.md` - Quick start training guide
- `notebooks/kaggle_dataset_integration.ipynb` - Dataset processing notebook
- `notebooks/medical_model_training.ipynb` - Model training notebook

## Future Enhancements

1. **Additional Data Sources**: PubMed, medical journals
2. **Multilingual Support**: Multiple language processing
3. **Real-time Updates**: Live data synchronization
4. **Advanced Guardrails**: More sophisticated safety checks
5. **User Interface**: Web-based chat interface
6. **Model Improvements**: Advanced training techniques, larger datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team.