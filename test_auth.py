#!/usr/bin/env python3
"""
Test script for Medical Knowledge RAG Chatbot Authentication
"""

import requests
import json

def test_authentication():
    """Test the authentication system"""
    base_url = "http://localhost:8000"
    
    print("🔐 MEDICAL CHATBOT AUTHENTICATION TEST")
    print("=" * 50)
    
    # Test 1: Login with valid credentials
    print("\n1️⃣ Testing Login...")
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        response = requests.post(f"{base_url}/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            print(f"✅ Login successful!")
            print(f"   Token: {access_token[:20]}...")
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Login request failed: {e}")
        return
    
    # Test 2: Get user info
    print("\n2️⃣ Testing User Info...")
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(f"{base_url}/auth/me", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ User info retrieved!")
            print(f"   Username: {user_data['username']}")
            print(f"   Email: {user_data['email']}")
            print(f"   Full Name: {user_data['full_name']}")
        else:
            print(f"❌ User info failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ User info request failed: {e}")
    
    # Test 3: Access protected endpoint (symptoms)
    print("\n3️⃣ Testing Protected Endpoint (Symptoms)...")
    
    try:
        response = requests.get(f"{base_url}/symptoms", headers=headers)
        if response.status_code == 200:
            symptoms_data = response.json()
            print(f"✅ Symptoms endpoint accessed!")
            print(f"   Available symptoms: {len(symptoms_data['symptoms'])}")
            print(f"   Sample symptoms: {symptoms_data['symptoms'][:5]}")
        else:
            print(f"❌ Symptoms endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Symptoms request failed: {e}")
    
    # Test 4: Access protected endpoint (disease prediction)
    print("\n4️⃣ Testing Protected Endpoint (Disease Prediction)...")
    
    prediction_data = {
        "symptoms": ["fever", "cough", "headache"],
        "age": 30,
        "gender": "male"
    }
    
    try:
        response = requests.post(f"{base_url}/predict-disease", json=prediction_data, headers=headers)
        if response.status_code == 200:
            prediction_result = response.json()
            print(f"✅ Disease prediction successful!")
            print(f"   Predictions: {len(prediction_result['predictions'])}")
            if prediction_result['predictions']:
                top_prediction = prediction_result['predictions'][0]
                print(f"   Top prediction: {top_prediction['disease']} ({top_prediction['confidence']*100:.1f}%)")
        else:
            print(f"❌ Disease prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Disease prediction request failed: {e}")
    
    # Test 5: Test without authentication (should fail)
    print("\n5️⃣ Testing Without Authentication (Should Fail)...")
    
    try:
        response = requests.post(f"{base_url}/predict-disease", json=prediction_data)
        if response.status_code == 401:
            print(f"✅ Authentication required (as expected)")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Authentication test completed!")

def test_different_users():
    """Test different user roles"""
    base_url = "http://localhost:8000"
    
    print("\n👥 TESTING DIFFERENT USER ROLES")
    print("=" * 40)
    
    users = [
        {"username": "admin", "password": "admin123", "role": "Administrator"},
        {"username": "doctor", "password": "doctor123", "role": "Medical Professional"},
        {"username": "user", "password": "user123", "role": "Regular User"}
    ]
    
    for user in users:
        print(f"\n🔑 Testing {user['role']} ({user['username']})...")
        
        # Login
        login_data = {"username": user["username"], "password": user["password"]}
        try:
            response = requests.post(f"{base_url}/auth/login", json=login_data)
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data["access_token"]
                headers = {"Authorization": f"Bearer {access_token}"}
                
                # Test disease prediction
                prediction_data = {"symptoms": ["fever", "cough"]}
                response = requests.post(f"{base_url}/predict-disease", json=prediction_data, headers=headers)
                
                if response.status_code == 200:
                    print(f"   ✅ {user['role']} can access disease prediction")
                else:
                    print(f"   ❌ {user['role']} cannot access disease prediction")
            else:
                print(f"   ❌ Login failed for {user['role']}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed for {user['role']}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Medical Chatbot Authentication Tests")
    
    # Test basic authentication
    test_authentication()
    
    # Test different user roles
    test_different_users()
    
    print("\n📋 AUTHENTICATION FEATURES:")
    print("✅ JWT-based authentication")
    print("✅ Protected endpoints")
    print("✅ User role management")
    print("✅ Token expiration (30 minutes)")
    print("✅ Secure API access")
    
    print("\n🔐 AVAILABLE USERS:")
    print("   • admin / admin123 (Administrator)")
    print("   • doctor / doctor123 (Medical Professional)")
    print("   • user / user123 (Regular User)")
    
    print("\n📡 API ENDPOINTS:")
    print("   • POST /auth/login - User login")
    print("   • GET /auth/me - Get user info")
    print("   • POST /auth/register - User registration")
    print("   • POST /predict-disease - Disease prediction (protected)")
    print("   • GET /symptoms - Get symptoms (protected)")
