"""
Medical validation utilities for safety and accuracy
"""
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of medical validation"""
    is_safe: bool
    confidence: float
    warnings: List[str]
    suggestions: List[str]
    risk_level: str


class MedicalValidator:
    """Validate medical content for safety and accuracy"""
    
    def __init__(self):
        self.dangerous_keywords = [
            "definitely", "certainly", "always", "never", "guaranteed",
            "cure", "heal", "treat", "diagnose", "prescribe"
        ]
        
        self.emergency_keywords = [
            "emergency", "urgent", "call 911", "ambulance", "hospital",
            "severe pain", "chest pain", "difficulty breathing", "unconscious"
        ]
        
        self.medical_disclaimers = [
            "This information is for educational purposes only",
            "Please consult a healthcare professional",
            "This should not replace medical advice"
        ]
    
    def validate_response(self, response: str, sources: List[Dict[str, Any]] = None) -> ValidationResult:
        """Validate medical response for safety"""
        warnings = []
        suggestions = []
        risk_level = "low"
        confidence = 1.0
        
        # Check for dangerous language
        for keyword in self.dangerous_keywords:
            if keyword.lower() in response.lower():
                warnings.append(f"Potentially unsafe language: '{keyword}'")
                risk_level = "medium"
                confidence -= 0.2
        
        # Check for emergency situations
        for keyword in self.emergency_keywords:
            if keyword.lower() in response.lower():
                warnings.append(f"Emergency situation mentioned: '{keyword}'")
                risk_level = "high"
                suggestions.append("Consider adding emergency contact information")
                confidence -= 0.3
        
        # Check source quality
        if sources:
            credible_sources = ["FDA", "WHO", "ClinicalTrials.gov", "PubMed"]
            has_credible = any(
                any(credible in source.get("source", "") for credible in credible_sources)
                for source in sources
            )
            
            if not has_credible:
                warnings.append("No credible medical sources found")
                confidence -= 0.1
        
        # Check for disclaimers
        has_disclaimer = any(
            disclaimer.lower() in response.lower() 
            for disclaimer in self.medical_disclaimers
        )
        
        if not has_disclaimer:
            suggestions.append("Consider adding medical disclaimer")
        
        # Determine safety
        is_safe = risk_level in ["low", "medium"] and confidence > 0.5
        
        return ValidationResult(
            is_safe=is_safe,
            confidence=max(0.0, confidence),
            warnings=warnings,
            suggestions=suggestions,
            risk_level=risk_level
        )
    
    def validate_query(self, query: str) -> ValidationResult:
        """Validate medical query for appropriateness"""
        warnings = []
        suggestions = []
        risk_level = "low"
        confidence = 1.0
        
        # Check for emergency keywords
        for keyword in self.emergency_keywords:
            if keyword.lower() in query.lower():
                warnings.append(f"Emergency keyword detected: '{keyword}'")
                risk_level = "high"
                suggestions.append("Please seek immediate medical attention")
                confidence -= 0.5
        
        # Check for inappropriate requests
        inappropriate_patterns = [
            r"diagnose\s+me",
            r"prescribe\s+me",
            r"tell\s+me\s+what\s+to\s+take",
            r"should\s+i\s+take"
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                warnings.append("Query requests medical advice")
                risk_level = "medium"
                suggestions.append("Please consult a healthcare professional")
                confidence -= 0.3
        
        # Check query quality
        if len(query) < 10:
            warnings.append("Query is too short")
            suggestions.append("Please provide more details")
            confidence -= 0.1
        
        if len(query) > 500:
            warnings.append("Query is too long")
            suggestions.append("Please simplify your question")
            confidence -= 0.1
        
        # Determine safety
        is_safe = risk_level in ["low", "medium"] and confidence > 0.5
        
        return ValidationResult(
            is_safe=is_safe,
            confidence=max(0.0, confidence),
            warnings=warnings,
            suggestions=suggestions,
            risk_level=risk_level
        )
    
    def add_safety_disclaimers(self, response: str) -> str:
        """Add appropriate medical disclaimers"""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare professional for medical decisions."
        return response + disclaimer
