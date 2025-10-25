"""
Medical-specific model definitions and architectures
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MedicalEmbeddingModel(nn.Module):
    """Medical-specific embedding model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for embedding generation"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode list of texts to embeddings"""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            embeddings = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return embeddings


class MedicalClassifier(nn.Module):
    """Medical text classifier"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class MedicalSafetyModel(nn.Module):
    """Model for medical safety classification"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.safety_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 classes: safe, warning, dangerous
        )
    
    def forward(self, x):
        return self.safety_classifier(x)


class MedicalRAGModel(nn.Module):
    """Combined RAG model for medical knowledge"""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generator_model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__()
        
        # Embedding model for retrieval
        self.embedding_model = MedicalEmbeddingModel(embedding_model_name)
        
        # Generator model for response generation
        self.generator_model = AutoModel.from_pretrained(generator_model_name)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        
        # Safety model
        self.safety_model = MedicalSafetyModel(384)  # MiniLM embedding size
        
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query for retrieval"""
        return self.embedding_model.encode_text([query])
    
    def encode_documents(self, documents: list) -> torch.Tensor:
        """Encode documents for retrieval"""
        return self.embedding_model.encode_text(documents)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using context"""
        # Implementation would depend on specific generator model
        # This is a placeholder for the actual generation logic
        pass
    
    def check_safety(self, text: str) -> Dict[str, Any]:
        """Check text for medical safety"""
        embedding = self.embedding_model.encode_text([text])
        safety_score = self.safety_model(embedding)
        
        return {
            "is_safe": safety_score.argmax().item() == 0,
            "confidence": torch.softmax(safety_score, dim=-1).max().item(),
            "risk_level": ["safe", "warning", "dangerous"][safety_score.argmax().item()]
        }
