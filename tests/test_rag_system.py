"""
Test cases for the Medical RAG system
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.vector_store import MedicalVectorStore
from rag.rag_engine import MedicalRAGEngine
from evaluation.medical_guardrails import MedicalGuardrails
from evaluation.evaluator import MedicalRAGEvaluator


class TestMedicalRAGSystem:
    """Test cases for the medical RAG system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.vector_store = MedicalVectorStore()
        self.rag_engine = MedicalRAGEngine(self.vector_store)
        self.guardrails = MedicalGuardrails()
        self.evaluator = MedicalRAGEvaluator()
    
    def test_vector_store_initialization(self):
        """Test vector store initialization"""
        assert self.vector_store is not None
        assert self.vector_store.chroma_client is not None
        assert self.vector_store.collection is not None
    
    def test_rag_engine_initialization(self):
        """Test RAG engine initialization"""
        assert self.rag_engine is not None
        assert self.rag_engine.vector_store is not None
    
    def test_guardrails_initialization(self):
        """Test guardrails initialization"""
        assert self.guardrails is not None
        assert self.guardrails.safety_threshold > 0
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        assert self.evaluator is not None
        assert self.evaluator.rouge_scorer is not None
    
    def test_medical_guardrails_safety_check(self):
        """Test medical guardrails safety check"""
        # Test safe response
        safe_response = "This medication is commonly used for diabetes treatment."
        sources = [{"source": "FDA", "similarity": 0.8}]
        
        validation = self.guardrails.validate_response(safe_response, sources)
        assert validation["is_safe"] is True
        
        # Test unsafe response
        unsafe_response = "You should definitely take this medication without consulting a doctor."
        validation = self.guardrails.validate_response(unsafe_response, sources)
        assert validation["is_safe"] is False
    
    def test_guardrails_emergency_detection(self):
        """Test emergency situation detection"""
        emergency_response = "If you experience chest pain, call 911 immediately."
        sources = [{"source": "FDA", "similarity": 0.9}]
        
        validation = self.guardrails.validate_response(emergency_response, sources)
        assert len(validation["suggestions"]) > 0
    
    def test_evaluator_rouge_scores(self):
        """Test ROUGE score calculation"""
        generated = "Metformin is used to treat diabetes."
        reference = "Metformin treats diabetes."
        
        rouge_scores = self.evaluator._calculate_rouge_scores(generated, reference)
        assert "rouge1" in rouge_scores
        assert "rouge2" in rouge_scores
        assert "rougeL" in rouge_scores
        assert all(0 <= score <= 1 for score in rouge_scores.values())
    
    def test_evaluator_bleu_score(self):
        """Test BLEU score calculation"""
        generated = "Metformin is used to treat diabetes."
        reference = "Metformin treats diabetes."
        
        bleu_score = self.evaluator._calculate_bleu_score(generated, reference)
        assert 0 <= bleu_score <= 1
    
    def test_evaluator_faithfulness(self):
        """Test faithfulness calculation"""
        response = "Metformin is used to treat diabetes."
        sources = [
            {"content": "Metformin is an antidiabetic medication", "similarity": 0.9},
            {"content": "Diabetes treatment includes metformin", "similarity": 0.8}
        ]
        
        faithfulness = self.evaluator._calculate_faithfulness(response, sources)
        assert 0 <= faithfulness <= 1
    
    def test_guardrails_query_filtering(self):
        """Test unsafe query filtering"""
        # Test safe query
        safe_query = "What is metformin used for?"
        is_safe, message = self.guardrails.filter_unsafe_queries(safe_query)
        assert is_safe is True
        
        # Test unsafe query
        unsafe_query = "Should I take metformin for my diabetes?"
        is_safe, message = self.guardrails.filter_unsafe_queries(unsafe_query)
        assert is_safe is False
        assert "medical advice" in message.lower()
    
    def test_evaluator_batch_evaluation(self):
        """Test batch evaluation"""
        test_cases = [
            {
                "query": "What is metformin?",
                "generated_response": "Metformin is an antidiabetic medication.",
                "reference_response": "Metformin is a diabetes medication.",
                "sources": [{"source": "FDA", "similarity": 0.9}]
            }
        ]
        
        results = self.evaluator.evaluate_batch(test_cases)
        assert "total_cases" in results
        assert "average_score" in results
        assert results["total_cases"] == 1
    
    def test_vector_store_search(self):
        """Test vector store search functionality"""
        # This test requires documents to be in the vector store
        # In a real test environment, you would add test documents first
        query = "diabetes medication"
        results = self.vector_store.search(query, top_k=3)
        
        # Results should be a list
        assert isinstance(results, list)
        
        # If results exist, they should have the expected structure
        if results:
            result = results[0]
            assert "content" in result
            assert "metadata" in result
            assert "similarity" in result
    
    def test_rag_engine_query_structure(self):
        """Test RAG engine query response structure"""
        # This test checks the structure of the response
        # In a real test, you would need documents in the vector store
        query = "What is metformin?"
        
        # Mock the vector store search to return empty results
        original_search = self.rag_engine.vector_store.search
        self.rag_engine.vector_store.search = lambda q, top_k=None: []
        
        try:
            result = self.rag_engine.query(query)
            
            # Check response structure
            assert "response" in result
            assert "sources" in result
            assert "context_used" in result
            assert "query" in result
            
            # Check types
            assert isinstance(result["response"], str)
            assert isinstance(result["sources"], list)
            assert isinstance(result["context_used"], int)
            assert isinstance(result["query"], str)
            
        finally:
            # Restore original method
            self.rag_engine.vector_store.search = original_search


if __name__ == "__main__":
    pytest.main([__file__])
