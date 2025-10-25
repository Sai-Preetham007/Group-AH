"""
Helper utilities for Medical Knowledge RAG Chatbot
"""
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-.,;:!?()]', '', text)
    
    # Normalize medical abbreviations
    medical_abbrevs = {
        'mg': 'milligrams',
        'ml': 'milliliters',
        'mcg': 'micrograms',
        'iu': 'international units',
        'bpm': 'beats per minute',
        'bp': 'blood pressure'
    }
    
    for abbrev, full_form in medical_abbrevs.items():
        text = re.sub(rf'\b{abbrev}\b', full_form, text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_medical_entities(text: str) -> List[str]:
    """Extract medical entities from text"""
    # Common medical terms and conditions
    medical_patterns = [
        r'\b(?:diabetes|hypertension|asthma|pneumonia|cancer|stroke|heart attack)\b',
        r'\b(?:mg|ml|mcg|iu|bpm|bp)\b',
        r'\b(?:symptoms|treatment|diagnosis|medication|therapy)\b',
        r'\b(?:doctor|physician|nurse|patient|hospital|clinic)\b'
    ]
    
    entities = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend(matches)
    
    return list(set(entities))


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity"""
    if not text1 or not text2:
        return 0.0
    
    # Simple Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def generate_content_hash(content: str) -> str:
    """Generate hash for content"""
    return hashlib.md5(content.encode()).hexdigest()


def validate_medical_query(query: str) -> Dict[str, Any]:
    """Validate medical query for safety and appropriateness"""
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "suggestions": [],
        "risk_level": "low"
    }
    
    # Check for emergency keywords
    emergency_keywords = [
        "emergency", "urgent", "call 911", "ambulance", "hospital",
        "severe pain", "chest pain", "difficulty breathing", "unconscious"
    ]
    
    for keyword in emergency_keywords:
        if keyword.lower() in query.lower():
            validation_result["warnings"].append(f"Emergency keyword detected: {keyword}")
            validation_result["risk_level"] = "high"
            validation_result["suggestions"].append("Please seek immediate medical attention")
    
    # Check for inappropriate requests
    inappropriate_patterns = [
        r"diagnose\s+me",
        r"prescribe\s+me",
        r"tell\s+me\s+what\s+to\s+take",
        r"should\s+i\s+take"
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            validation_result["warnings"].append("Query requests medical advice")
            validation_result["risk_level"] = "medium"
            validation_result["suggestions"].append("Please consult a healthcare professional")
    
    # Check query length
    if len(query) < 10:
        validation_result["warnings"].append("Query is too short")
        validation_result["suggestions"].append("Please provide more details")
    
    if len(query) > 500:
        validation_result["warnings"].append("Query is too long")
        validation_result["suggestions"].append("Please simplify your question")
    
    return validation_result


def format_medical_response(response: str, sources: List[Dict[str, Any]] = None) -> str:
    """Format medical response with proper disclaimers"""
    
    # Add medical disclaimer
    disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare professional for medical decisions."
    
    # Add source citations if available
    if sources:
        source_text = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            source_name = source.get('source', 'Unknown')
            source_id = source.get('id', '')
            source_text += f"{i}. {source_name}"
            if source_id:
                source_text += f" (ID: {source_id})"
            source_text += "\n"
        
        response += source_text
    
    response += disclaimer
    return response


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    path = Path(filepath)
    
    if not path.exists():
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks


def merge_duplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate sources based on content"""
    seen = set()
    merged_sources = []
    
    for source in sources:
        content = source.get('content', '')
        content_hash = generate_content_hash(content)
        
        if content_hash not in seen:
            seen.add(content_hash)
            merged_sources.append(source)
    
    return merged_sources
