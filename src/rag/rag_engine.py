"""
RAG Engine for Medical Knowledge Chatbot
"""
import openai
from typing import List, Dict, Any, Optional
import logging
from .vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


class MedicalRAGEngine:
    """RAG Engine for medical knowledge retrieval and generation"""
    
    def __init__(self, vector_store: MedicalVectorStore, llm_model: str = "gpt-3.5-turbo"):
        self.vector_store = vector_store
        self.llm_model = llm_model
        
        # Initialize OpenAI client
        # Note: Set OPENAI_API_KEY environment variable
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        logger.warning("OpenAI API key not provided")
    
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
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query method that combines retrieval and generation"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_documents(question)
            
            if not retrieved_docs:
                return {
                    "response": "I couldn't find relevant medical information for your query. Please try rephrasing your question or consult a healthcare professional.",
                    "sources": [],
                    "context_used": 0,
                    "query": question
                }
            
            # Generate response
            result = self.generate_response(question, retrieved_docs)
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
            if not settings.openai_api_key:
                return "Error: OpenAI API key not configured"
            
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
            return f"Error generating response: {str(e)}"
    
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
