#!/usr/bin/env python3
"""
Test script for WHO and openFDA source verification in RAG engine
"""
import sys
sys.path.append('src')

from src.rag.rag_engine import MedicalRAGEngine
from src.rag.vector_store import MedicalVectorStore

def test_medical_source_verification():
    """Test WHO and openFDA source verification"""
    print("🔍 TESTING MEDICAL SOURCE VERIFICATION")
    print("=" * 60)
    
    try:
        # Initialize RAG engine (without vector store for this test)
        rag_engine = MedicalRAGEngine(None)
        
        # Test queries
        test_queries = [
            {
                "query": "What are the symptoms of malaria?",
                "disease_name": "malaria",
                "drug_name": None
            },
            {
                "query": "What are the side effects of aspirin?",
                "disease_name": None,
                "drug_name": "aspirin"
            },
            {
                "query": "How to treat diabetes?",
                "disease_name": "diabetes",
                "drug_name": None
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🔬 Test {i}: {test_case['query']}")
            print("-" * 40)
            
            # Test WHO verification
            if test_case['disease_name']:
                print("🌍 Testing WHO verification...")
                who_result = rag_engine.verify_with_who(
                    test_case['query'], 
                    test_case['disease_name']
                )
                print(f"   WHO Verified: {who_result['verified']}")
                print(f"   Sources: {len(who_result['sources'])}")
            
            # Test openFDA verification
            if test_case['drug_name']:
                print("💊 Testing openFDA verification...")
                fda_result = rag_engine.verify_with_fda(
                    test_case['query'], 
                    test_case['drug_name']
                )
                print(f"   FDA Verified: {fda_result['verified']}")
                print(f"   Sources: {len(fda_result['sources'])}")
            
            # Test combined verification
            print("🔗 Testing combined verification...")
            combined_result = rag_engine.verify_medical_sources(
                test_case['query'],
                test_case['disease_name'],
                test_case['drug_name']
            )
            print(f"   Overall Verified: {combined_result['overall_verified']}")
            print(f"   Verified Sources: {combined_result['verified_sources']}")
        
        print("\n✅ Medical source verification testing complete!")
        
    except Exception as e:
        print(f"❌ Error testing medical sources: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_medical_source_verification()
