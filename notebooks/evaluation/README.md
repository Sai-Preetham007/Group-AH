# Evaluation Notebooks

This directory contains notebooks for evaluating the Medical Knowledge RAG Chatbot system.

## Notebooks

### 1. `model_evaluation.ipynb`
Comprehensive model evaluation including:
- Model performance metrics
- Confusion matrix analysis
- Disease-wise accuracy
- Classification reports
- Performance visualizations

### 2. `rag_system_evaluation.ipynb`
End-to-end RAG system evaluation:
- API endpoint testing
- Response quality assessment
- Medical safety evaluation
- Performance benchmarking
- User experience testing

### 3. `comparative_analysis.ipynb`
Comparative analysis of different models:
- Model comparison
- Performance trade-offs
- Cost-benefit analysis
- Recommendation engine

## Evaluation Metrics

### Model Performance
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### RAG System Performance
- **Response Time**: Average time to generate response
- **Relevance**: How relevant the response is to the query
- **Accuracy**: Medical accuracy of responses
- **Safety**: Medical safety compliance

### Medical Safety
- **Emergency Detection**: Ability to detect emergency situations
- **Disclaimer Usage**: Proper use of medical disclaimers
- **Source Citation**: Accuracy of source citations
- **Risk Assessment**: Proper risk level assessment

## Usage

### Running Model Evaluation
```bash
# Start Jupyter notebook
jupyter notebook notebooks/evaluation/

# Run model_evaluation.ipynb
# This will evaluate the trained model on test data
```

### Running RAG System Evaluation
```bash
# Make sure the API server is running
python run_server.py

# Run rag_system_evaluation.ipynb
# This will test the complete RAG system
```

### Running Comparative Analysis
```bash
# Run comparative_analysis.ipynb
# This will compare different models and approaches
```

## Evaluation Results

Results are saved to `data/evaluation/` directory:
- `model_evaluation.json`: Model performance results
- `rag_system_evaluation.json`: RAG system results
- `comparative_analysis.json`: Comparative analysis results

## Best Practices

1. **Run evaluations regularly** to monitor model performance
2. **Use diverse test cases** to ensure robust evaluation
3. **Compare with baselines** to understand improvement
4. **Document evaluation results** for future reference
5. **Monitor for model drift** over time
