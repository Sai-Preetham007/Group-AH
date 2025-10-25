"""
Vector store implementation for medical knowledge retrieval
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """Vector store for medical knowledge retrieval"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chroma_client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure directory exists
            persist_dir = Path(settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"description": "Medical knowledge base for RAG system"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        try:
            # Extract content and metadata
            contents = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                contents.append(doc["content"])
                metadata = {k: v for k, v in doc.items() if k != "content"}
                metadatas.append(metadata)
                ids.append(f"doc_{i}")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if top_k is None:
                top_k = settings.top_k_retrieval
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                search_results.append(result)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result["similarity"] >= settings.similarity_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)"""
        try:
            self.chroma_client.delete_collection("medical_knowledge")
            self.collection = self.chroma_client.create_collection(
                name="medical_knowledge",
                metadata={"description": "Medical knowledge base for RAG system"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
