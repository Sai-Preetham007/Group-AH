#!/usr/bin/env python3
"""
Simple API Login Test Script
Test the authentication endpoints of the Medical Knowledge RAG Chatbot
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_login(username, password):
    """Test login with given credentials"""
    print(f"\n🔐 Testing login for: {username}")
    print("-" * 40)
    
    try:
        # Login request
        login_data = {
            "username": username,
            "password": password
        }
        
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json=login_data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data["access_token"]
            print("✅ Login successful!")
            print(f"Token: {token[:30]}...")
            
            # Test token by accessing protected endpoint
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test /auth/me endpoint
            me_response = requests.get(f"{BASE_URL}/auth/me", headers=headers, timeout=10)
            if me_response.status_code == 200:
                user_info = me_response.json()
                print(f"✅ Token validation successful!")
                print(f"User: {user_info['username']}")
                print(f"Email: {user_info['email']}")
                print(f"Full Name: {user_info['full_name']}")
                return token
            else:
                print(f"❌ Token validation failed: {me_response.status_code}")
                print(f"Error: {me_response.text}")
                return None
        else:
            print(f"❌ Login failed!")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Is the server running?")
        print("   Start server with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_disease_prediction(token):
    """Test disease prediction with authentication"""
    if not token:
        print("❌ No token available for disease prediction test")
        return
    
    print(f"\n🏥 Testing Disease Prediction")
    print("-" * 40)
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test symptoms endpoint
        symptoms_response = requests.get(f"{BASE_URL}/symptoms", headers=headers, timeout=10)
        if symptoms_response.status_code == 200:
            symptoms_data = symptoms_response.json()
            print(f"✅ Symptoms endpoint: {len(symptoms_data['symptoms'])} symptoms available")
        else:
            print(f"❌ Symptoms endpoint failed: {symptoms_response.status_code}")
            return
        
        # Test disease prediction
        prediction_data = {
            "symptoms": ["fever", "cough", "headache"],
            "age": 30,
            "gender": "male"
        }
        
        prediction_response = requests.post(
            f"{BASE_URL}/predict-disease",
            json=prediction_data,
            headers=headers,
            timeout=10
        )
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            print(f"✅ Disease prediction successful!")
            print(f"Analysis: {result['analysis_summary']}")
            
            if result['predictions']:
                top_pred = result['predictions'][0]
                print(f"Top prediction: {top_pred['disease']} ({top_pred['confidence']*100:.1f}%)")
                print(f"Severity: {top_pred['severity_score']}/10")
                print(f"Urgency: {top_pred['urgency_level']}")
        else:
            print(f"❌ Disease prediction failed: {prediction_response.status_code}")
            print(f"Error: {prediction_response.text}")
            
    except Exception as e:
        print(f"❌ Disease prediction error: {e}")

def main():
    """Main test function"""
    print("🚀 Medical Knowledge RAG Chatbot - API Login Test")
    print("=" * 60)
    
    # Test server health first
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("   Make sure the server is running on http://localhost:8000")
        return
    
    # Test different user logins
    test_users = [
        ("admin", "admin123"),
        ("doctor", "doctor123"), 
        ("user", "user123")
    ]
    
    successful_logins = 0
    
    for username, password in test_users:
        token = test_login(username, password)
        if token:
            successful_logins += 1
            # Test disease prediction with the first successful login
            if successful_logins == 1:
                test_disease_prediction(token)
    
    print(f"\n📊 Test Results:")
    print(f"   Successful logins: {successful_logins}/{len(test_users)}")
    
    if successful_logins > 0:
        print("✅ Authentication system is working correctly!")
    else:
        print("❌ Authentication system has issues")
        print("\n🔧 Troubleshooting:")
        print("1. Check if server is running: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        print("2. Check server logs for errors")
        print("3. Verify user credentials in src/api/auth.py")

if __name__ == "__main__":
    main()
