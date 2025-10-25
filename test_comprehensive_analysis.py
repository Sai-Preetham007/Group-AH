#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive disease analysis
"""

import requests
import json
from typing import Dict, Any

def test_comprehensive_analysis():
    """Test the comprehensive disease analysis API"""
    
    base_url = "http://localhost:8000"
    
    # Test cases with different scenarios
    test_cases = [
        {
            "name": "Common Cold Case",
            "symptoms": ["fever", "cough", "headache"],
            "age": 25,
            "gender": "female"
        },
        {
            "name": "Serious Cardiac Case",
            "symptoms": ["chest pain", "shortness of breath", "dizziness"],
            "age": 55,
            "gender": "male"
        },
        {
            "name": "Elderly High Risk Case",
            "symptoms": ["fever", "confusion", "weakness"],
            "age": 75,
            "gender": "female"
        },
        {
            "name": "Child Case",
            "symptoms": ["fever", "rash", "irritability"],
            "age": 3,
            "gender": "male"
        }
    ]
    
    print("🏥 COMPREHENSIVE DISEASE ANALYSIS TEST")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 TEST CASE {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Make API request
            response = requests.post(
                f"{base_url}/predict-disease",
                json={
                    "symptoms": test_case["symptoms"],
                    "age": test_case["age"],
                    "gender": test_case["gender"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display comprehensive analysis
                print(f"🔍 INPUT:")
                print(f"   Symptoms: {', '.join(test_case['symptoms'])}")
                print(f"   Age: {test_case['age']}, Gender: {test_case['gender']}")
                
                print(f"\n📊 ANALYSIS SUMMARY:")
                print(f"   {data['analysis_summary']}")
                
                print(f"\n🎯 TOP PREDICTIONS:")
                for j, prediction in enumerate(data['predictions'][:2], 1):
                    print(f"\n   {j}. {prediction['disease']}")
                    print(f"      Confidence: {prediction['confidence']*100:.1f}%")
                    print(f"      Severity Score: {prediction['severity_score']}/10")
                    print(f"      Urgency Level: {prediction['urgency_level']}")
                    
                    if prediction['description']:
                        print(f"      Description: {prediction['description'][:100]}...")
                    
                    print(f"\n      🚨 RISK FACTORS:")
                    for risk in prediction['risk_factors']:
                        print(f"         • {risk['factor']} ({risk['level']}): {risk['description']}")
                    
                    print(f"\n      💡 LLM-GENERATED PRECAUTIONS:")
                    for k, precaution in enumerate(prediction['precautions'][:5], 1):
                        print(f"         {k}. {precaution}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        print("\n" + "=" * 60)

def test_llm_precautions():
    """Test LLM-generated precautions for different scenarios"""
    
    print("\n🤖 LLM-GENERATED PRECAUTIONS TEST")
    print("=" * 50)
    
    scenarios = [
        {
            "disease": "Malaria",
            "symptoms": ["fever", "chills", "sweating"],
            "urgency": "Emergency"
        },
        {
            "disease": "Diabetes",
            "symptoms": ["excessive thirst", "frequent urination"],
            "urgency": "High"
        },
        {
            "disease": "Common Cold",
            "symptoms": ["runny nose", "sneezing"],
            "urgency": "Low"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 SCENARIO: {scenario['disease']}")
        print(f"   Symptoms: {', '.join(scenario['symptoms'])}")
        print(f"   Urgency: {scenario['urgency']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/predict-disease",
                json={"symptoms": scenario["symptoms"]},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['predictions']:
                    prediction = data['predictions'][0]
                    print(f"\n   🎯 PREDICTED: {prediction['disease']}")
                    print(f"   📊 CONFIDENCE: {prediction['confidence']*100:.1f}%")
                    print(f"   ⚠️  URGENCY: {prediction['urgency_level']}")
                    print(f"   📈 SEVERITY: {prediction['severity_score']}/10")
                    
                    print(f"\n   💡 LLM PRECAUTIONS:")
                    for i, precaution in enumerate(prediction['precautions'], 1):
                        print(f"      {i}. {precaution}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Disease Analysis")
    print("=" * 60)
    
    # Test comprehensive analysis
    test_comprehensive_analysis()
    
    # Test LLM precautions
    test_llm_precautions()
    
    print("\n✅ Testing completed!")
    print("\n📋 SUMMARY OF FEATURES:")
    print("1. ✅ Disease prediction with confidence scores")
    print("2. ✅ Risk factor analysis (age, gender, symptoms)")
    print("3. ✅ Severity scoring (0-10 scale)")
    print("4. ✅ Urgency level assessment (Low/Medium/High/Emergency)")
    print("5. ✅ LLM-generated precautions")
    print("6. ✅ Comprehensive analysis summary")
    print("7. ✅ Medical descriptions from trained dataset")
    print("8. ✅ Age and gender-based risk assessment")
