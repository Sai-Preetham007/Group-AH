"""
Database configuration and models for Medical Knowledge RAG Chatbot
"""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import hashlib

# Simple password hashing functions
def hash_password(password: str) -> str:
    """Simple password hashing using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

# Database URL - using SQLite for simplicity, can be changed to PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_chatbot.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

# Disease prediction history model
class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    symptoms = Column(Text, nullable=False)  # JSON string of symptoms
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(String(10), nullable=False)  # Store as string for precision
    severity_score = Column(String(10), nullable=False)
    urgency_level = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PredictionHistory(user_id={self.user_id}, disease='{self.predicted_disease}')>"

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database with default admin user
def init_database():
    """Initialize database with default admin user"""
    
    # Create tables
    create_tables()
    
    # Create default admin user if not exists
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            # Create admin user
            admin_user = User(
                username="admin",
                email="admin@medicalchatbot.com",
                full_name="System Administrator",
                hashed_password=hash_password("admin123"),
                is_active=True,
                is_admin=True
            )
            db.add(admin_user)
            
            # Add default users
            doctor_user = User(
                username="doctor",
                email="doctor@medicalchatbot.com",
                full_name="Dr. Medical Professional",
                hashed_password=hash_password("doctor123"),
                is_active=True,
                is_admin=False
            )
            db.add(doctor_user)
            
            regular_user = User(
                username="user",
                email="user@medicalchatbot.com",
                full_name="Regular User",
                hashed_password=hash_password("user123"),
                is_active=True,
                is_admin=False
            )
            db.add(regular_user)
            
            db.commit()
            print("✅ Database initialized with default users")
        else:
            print("✅ Database already initialized")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()
