"""
Medical guardrails for safe and accurate responses
"""
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedicalGuardrails:
    """Guardrails for medical safety and accuracy"""
    
    def __init__(self, safety_threshold: float = 0.8):
        self.safety_threshold = safety_threshold
        self.medical_disclaimers = [
            "This information is for educational purposes only and should not replace professional medical advice.",
            "Please consult with a healthcare professional for medical decisions.",
            "This response is based on available medical literature and should be verified with current guidelines."
        ]
        
        # Medical safety keywords
        self.dangerous_keywords = [
            "diagnose", "prescribe", "treatment plan", "medical advice",
            "you should take", "you must", "definitely", "certainly"
        ]
        
        # Emergency keywords
        self.emergency_keywords = [
            "emergency", "urgent", "call 911", "ambulance", "hospital",
            "severe pain", "chest pain", "difficulty breathing"
        ]
    
    def validate_response(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate medical response for safety and accuracy"""
        try:
            validation_result = {
                "is_safe": True,
                "warnings": [],
                "suggestions": [],
                "confidence_score": 0.0,
                "requires_disclaimer": False
            }
            
            # Check for dangerous language
            safety_check = self._check_safety(response)
            validation_result.update(safety_check)
            
            # Check for emergency situations
            emergency_check = self._check_emergency(response)
            validation_result.update(emergency_check)
            
            # Check source quality
            source_check = self._check_sources(sources)
            validation_result.update(source_check)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(response, sources)
            validation_result["confidence_score"] = confidence
            
            # Determine if disclaimers are needed
            validation_result["requires_disclaimer"] = self._needs_disclaimer(response)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return {
                "is_safe": False,
                "warnings": [f"Validation error: {str(e)}"],
                "suggestions": ["Please try rephrasing your question"],
                "confidence_score": 0.0,
                "requires_disclaimer": True
            }
    
    def _check_safety(self, response: str) -> Dict[str, Any]:
        """Check response for potentially unsafe medical language"""
        warnings = []
        is_safe = True
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword.lower() in response.lower():
                warnings.append(f"Response contains potentially unsafe language: '{keyword}'")
                is_safe = False
        
        # Check for absolute statements
        absolute_patterns = [
            r"definitely\s+\w+",
            r"certainly\s+\w+",
            r"always\s+\w+",
            r"never\s+\w+"
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                warnings.append("Response contains absolute statements that may be inappropriate for medical advice")
                is_safe = False
        
        return {
            "is_safe": is_safe,
            "warnings": warnings
        }
    
    def _check_emergency(self, response: str) -> Dict[str, Any]:
        """Check if response addresses emergency situations"""
        suggestions = []
        
        for keyword in self.emergency_keywords:
            if keyword.lower() in response.lower():
                suggestions.append("Consider adding emergency contact information")
                break
        
        return {
            "suggestions": suggestions
        }
    
    def _check_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check quality and reliability of sources"""
        warnings = []
        suggestions = []
        
        if not sources:
            warnings.append("No sources provided for medical information")
            return {"warnings": warnings, "suggestions": suggestions}
        
        # Check source diversity
        source_types = set(source.get("source", "Unknown") for source in sources)
        if len(source_types) < 2:
            suggestions.append("Consider including multiple source types for better reliability")
        
        # Check source credibility
        credible_sources = ["FDA", "WHO", "ClinicalTrials.gov", "PubMed"]
        has_credible_source = any(
            any(credible in source.get("source", "") for credible in credible_sources)
            for source in sources
        )
        
        if not has_credible_source:
            warnings.append("No credible medical sources found in retrieved information")
        
        return {
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _calculate_confidence(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response"""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor in number of sources
            if sources:
                confidence += min(len(sources) * 0.1, 0.3)
            
            # Factor in source similarity scores
            if sources:
                avg_similarity = sum(source.get("similarity", 0) for source in sources) / len(sources)
                confidence += avg_similarity * 0.2
            
            # Factor in response length (more detailed responses tend to be more confident)
            if len(response) > 200:
                confidence += 0.1
            
            # Factor in presence of disclaimers (shows awareness of limitations)
            if any(disclaimer in response for disclaimer in self.medical_disclaimers):
                confidence += 0.1
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _needs_disclaimer(self, response: str) -> bool:
        """Determine if response needs medical disclaimers"""
        # Check if response contains medical information
        medical_indicators = [
            "drug", "medication", "treatment", "therapy", "disease", "condition",
            "symptom", "diagnosis", "dosage", "side effect", "contraindication"
        ]
        
        return any(indicator in response.lower() for indicator in medical_indicators)
    
    def add_safety_disclaimers(self, response: str) -> str:
        """Add appropriate medical disclaimers to response"""
        if not self._needs_disclaimer(response):
            return response
        
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: " + self.medical_disclaimers[0]
        return response + disclaimer
    
    def filter_unsafe_queries(self, query: str) -> Tuple[bool, str]:
        """Filter out potentially unsafe queries"""
        # Check for queries asking for specific medical advice
        unsafe_patterns = [
            r"should i take",
            r"can i take",
            r"is it safe to",
            r"what should i do for",
            r"how to treat",
            r"diagnose my"
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "I cannot provide specific medical advice. Please consult a healthcare professional."
        
        return True, query
