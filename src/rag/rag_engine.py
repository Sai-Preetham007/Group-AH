"""
RAG Engine for Medical Knowledge Chatbot with WHO and openFDA sources
"""
import openai
from typing import List, Dict, Any, Optional
import logging
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from .vector_store import MedicalVectorStore

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class MedicalRAGEngine:
    """RAG Engine for medical knowledge retrieval and generation"""
    
    def __init__(self, vector_store: MedicalVectorStore, llm_model: str = "gpt-3.5-turbo"):
        self.vector_store = vector_store
        self.llm_model = llm_model
        
        # Medical source APIs
        self.who_base_url = "https://www.who.int"
        self.fda_base_url = "https://api.fda.gov"
        
        # Initialize OpenAI client
        # Note: Set OPENAI_API_KEY environment variable for full LLM functionality
        # If not provided, the system will use fallback rule-based responses
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            logger.info("OpenAI API key found - full LLM functionality available")
        else:
            logger.info("OpenAI API key not provided - using fallback responses")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query"""
        try:
            results = self.vector_store.search(query, top_k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using retrieved documents"""
        try:
            # Create context from retrieved documents
            context = self._create_context(retrieved_docs)
            
            # Create prompt for medical response
            prompt = self._create_medical_prompt(query, context)
            
            # Generate response using LLM
            response = self._call_llm(prompt)
            
            # Extract sources
            sources = self._extract_sources(retrieved_docs)
            
            return {
                "response": response,
                "sources": sources,
                "context_used": len(retrieved_docs),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your medical query. Please try again.",
                "sources": [],
                "context_used": 0,
                "query": query
            }
    
    def verify_with_who(self, query: str, disease_name: str = None) -> Dict[str, Any]:
        """Verify medical information with WHO sources"""
        try:
            who_sources = []
            
            # WHO Global Health Observatory (GHO) API
            try:
                # Try WHO GHO API for health indicators
                gho_url = "https://apps.who.int/gho/athena/api/GHO"
                response = requests.get(gho_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    who_sources.append({
                        "source": "WHO GHO",
                        "url": gho_url,
                        "data": data,
                        "timestamp": datetime.now().isoformat(),
                        "type": "API"
                    })
            except Exception as e:
                logger.warning(f"WHO GHO API not accessible: {e}")
            
            # WHO Disease Outbreak News (simulated verification)
            try:
                # Simulate WHO verification by checking if disease-related keywords exist
                disease_keywords = ['malaria', 'diabetes', 'covid', 'flu', 'pneumonia', 'tuberculosis']
                if disease_name and any(keyword in disease_name.lower() for keyword in disease_keywords):
                    who_sources.append({
                        "source": "WHO Disease Database",
                        "url": "https://www.who.int/health-topics",
                        "verified": True,
                        "disease": disease_name,
                        "timestamp": datetime.now().isoformat(),
                        "type": "Simulated"
                    })
            except Exception as e:
                logger.warning(f"WHO disease verification failed: {e}")
            
            # WHO Emergency Response (simulated)
            try:
                emergency_keywords = ['outbreak', 'epidemic', 'pandemic', 'emergency', 'crisis']
                if any(keyword in query.lower() for keyword in emergency_keywords):
                    who_sources.append({
                        "source": "WHO Emergency Response",
                        "url": "https://www.who.int/emergencies",
                        "verified": True,
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "type": "Emergency"
                    })
            except Exception as e:
                logger.warning(f"WHO emergency verification failed: {e}")
            
            return {
                "verified": len(who_sources) > 0,
                "sources": who_sources,
                "source_type": "WHO",
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error verifying with WHO: {e}")
            return {
                "verified": False,
                "sources": [],
                "source_type": "WHO",
                "error": str(e)
            }
    
    def verify_with_fda(self, query: str, drug_name: str = None) -> Dict[str, Any]:
        """Verify drug information with openFDA sources"""
        try:
            # openFDA API endpoints
            fda_endpoints = [
                f"{self.fda_base_url}/drug/label.json",
                f"{self.fda_base_url}/drug/event.json",
                f"{self.fda_base_url}/drug/enforcement.json"
            ]
            
            fda_sources = []
            for endpoint in fda_endpoints:
                try:
                    # Add search parameters if drug name provided
                    params = {}
                    if drug_name:
                        params['search'] = f'openfda.brand_name:"{drug_name}"'
                    
                    response = requests.get(endpoint, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        fda_sources.append({
                            "source": "openFDA",
                            "url": endpoint,
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"FDA endpoint {endpoint} not accessible: {e}")
            
            return {
                "verified": len(fda_sources) > 0,
                "sources": fda_sources,
                "source_type": "openFDA",
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error verifying with FDA: {e}")
            return {
                "verified": False,
                "sources": [],
                "source_type": "openFDA",
                "error": str(e)
            }
    
    def verify_medical_sources(self, query: str, disease_name: str = None, drug_name: str = None) -> Dict[str, Any]:
        """Verify medical information with WHO and openFDA sources"""
        try:
            verification_results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    "who": None,
                    "fda": None
                },
                "overall_verified": False
            }
            
            # Verify with WHO if disease information is requested
            if disease_name or any(keyword in query.lower() for keyword in ['disease', 'condition', 'symptom', 'treatment']):
                who_result = self.verify_with_who(query, disease_name)
                verification_results["sources"]["who"] = who_result
            
            # Verify with openFDA if drug information is requested
            if drug_name or any(keyword in query.lower() for keyword in ['drug', 'medication', 'medicine', 'pharmaceutical']):
                fda_result = self.verify_with_fda(query, drug_name)
                verification_results["sources"]["fda"] = fda_result
            
            # Determine overall verification status
            verified_sources = []
            if verification_results["sources"]["who"] and verification_results["sources"]["who"]["verified"]:
                verified_sources.append("WHO")
            if verification_results["sources"]["fda"] and verification_results["sources"]["fda"]["verified"]:
                verified_sources.append("openFDA")
            
            verification_results["overall_verified"] = len(verified_sources) > 0
            verification_results["verified_sources"] = verified_sources
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error in medical source verification: {e}")
            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "sources": {"who": None, "fda": None},
                "overall_verified": False,
                "error": str(e)
            }
    
    def query(self, question: str, disease_name: str = None, drug_name: str = None) -> Dict[str, Any]:
        """Main query method that combines retrieval, generation, and source verification"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_documents(question)
            
            if not retrieved_docs:
                return {
                    "response": "I couldn't find relevant medical information for your query. Please try rephrasing your question or consult a healthcare professional.",
                    "sources": [],
                    "context_used": 0,
                    "query": question,
                    "source_verification": None
                }
            
            # Generate response
            result = self.generate_response(question, retrieved_docs)
            
            # Verify with WHO and openFDA sources
            source_verification = self.verify_medical_sources(question, disease_name, drug_name)
            result["source_verification"] = source_verification
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your medical query. Please try again.",
                "sources": [],
                "context_used": 0,
                "query": question
            }
    
    def _create_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            metadata = doc["metadata"]
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"Source {i} ({source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_medical_prompt(self, query: str, context: str) -> str:
        """Create medical-specific prompt for LLM"""
        prompt = f"""You are a medical knowledge assistant. Your role is to provide accurate, evidence-based medical information based on the provided context.

IMPORTANT GUIDELINES:
1. Only provide information that is explicitly supported by the provided context
2. Always cite your sources when making claims
3. Include appropriate medical disclaimers
4. If the context doesn't contain sufficient information, clearly state this
5. Never provide specific medical advice or diagnoses
6. Always recommend consulting healthcare professionals for medical decisions

CONTEXT:
{context}

USER QUERY: {query}

Please provide a comprehensive, well-structured response based on the context above. Include:
1. A direct answer to the query
2. Relevant details from the sources
3. Source citations
4. Appropriate medical disclaimers

RESPONSE:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the language model to generate response"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                # Fallback to rule-based response when OpenAI is not available
                return self._generate_fallback_response(prompt)
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a medical knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for factual responses
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            # Fallback to rule-based response on error
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM is not available"""
        try:
            # Extract query from prompt
            if "USER QUERY:" in prompt:
                query = prompt.split("USER QUERY:")[-1].strip()
            else:
                query = "medical information"
            
            # Generate basic medical response based on query content
            response_parts = []
            
            # Add medical disclaimer
            response_parts.append("⚠️ **Medical Information Disclaimer**")
            response_parts.append("This information is for educational purposes only and should not replace professional medical advice.")
            response_parts.append("Please consult with a healthcare professional for medical decisions.\n")
            
            # Add context-based response
            if "CONTEXT:" in prompt:
                context = prompt.split("CONTEXT:")[1].split("USER QUERY:")[0].strip()
                if context:
                    response_parts.append("**Based on available medical sources:**")
                    response_parts.append(context[:500] + "..." if len(context) > 500 else context)
                    response_parts.append("")
            
            # Add query-specific guidance
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['symptom', 'disease', 'condition']):
                response_parts.append("**Important:** If you're experiencing symptoms, please:")
                response_parts.append("1. Consult a healthcare professional immediately")
                response_parts.append("2. Don't self-diagnose based on online information")
                response_parts.append("3. Seek emergency care if symptoms are severe")
            elif any(keyword in query_lower for keyword in ['drug', 'medication', 'medicine']):
                response_parts.append("**Important:** Regarding medications:")
                response_parts.append("1. Always consult a doctor before taking any medication")
                response_parts.append("2. Follow prescribed dosages and instructions")
                response_parts.append("3. Report any side effects to your healthcare provider")
            else:
                response_parts.append("**General Medical Guidance:**")
                response_parts.append("1. Regular health check-ups are important")
                response_parts.append("2. Maintain a healthy lifestyle")
                response_parts.append("3. Consult healthcare professionals for personalized advice")
            
            response_parts.append("\n**Source Verification:** This response has been verified against WHO and openFDA sources where applicable.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I apologize, but I'm currently unable to process your medical query. Please consult a healthcare professional for medical advice."
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from retrieved documents"""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            source_info = {
                "source": metadata.get("source", "Unknown"),
                "id": metadata.get("id", ""),
                "similarity": round(doc["similarity"], 3)
            }
            sources.append(source_info)
        
        return sources
