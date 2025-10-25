"""
Authentication module for Medical Knowledge RAG Chatbot
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security scheme
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    full_name: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str
    full_name: str
    is_active: bool = True

# In-memory user storage (replace with database in production)
users_db = {
    "admin": {
        "username": "admin",
        "password": "admin123",  # In production, use hashed passwords
        "email": "admin@medicalchatbot.com",
        "full_name": "System Administrator",
        "is_active": True
    },
    "doctor": {
        "username": "doctor",
        "password": "doctor123",
        "email": "doctor@medicalchatbot.com", 
        "full_name": "Dr. Medical Professional",
        "is_active": True
    },
    "user": {
        "username": "user",
        "password": "user123",
        "email": "user@medicalchatbot.com",
        "full_name": "Regular User",
        "is_active": True
    }
}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(username: str = Depends(verify_token)):
    """Get current authenticated user"""
    if username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    user_data = users_db[username]
    return User(
        username=user_data["username"],
        email=user_data["email"],
        full_name=user_data["full_name"],
        is_active=user_data["is_active"]
    )

def authenticate_user(username: str, password: str):
    """Authenticate user with username and password"""
    user = users_db.get(username)
    if not user:
        return False
    if user["password"] != password:
        return False
    return user
