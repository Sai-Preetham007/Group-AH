"""
FastAPI main application for Medical Knowledge RAG Chatbot
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import logging
import uvicorn
import pandas as pd
import json
from pathlib import Path
from .auth_db import (
    UserLogin, UserRegister, Token, UserResponse, 
    create_access_token, get_current_user, authenticate_user, create_user,
    get_current_admin_user
)
from ..database import get_db, init_database, PredictionHistory
from sqlalchemy.orm import Session
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    yield
    # Cleanup code can go here if needed

# Pydantic models for symptom prediction
class SymptomInput(BaseModel):
    symptoms: List[str]
    age: Optional[int] = None
    gender: Optional[str] = None

class RiskFactor(BaseModel):
    factor: str
    level: str  # "Low", "Medium", "High"
    description: str

class DrugInfo(BaseModel):
    name: str
    type: str  # "Prescription", "Over-the-counter", "Supplement"

class TreatmentInfo(BaseModel):
    drugs: List[DrugInfo]
    treatments: List[str]
    duration: str
    notes: Optional[str] = None

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    description: Optional[str] = None
    precautions: Optional[List[str]] = None
    risk_factors: Optional[List[RiskFactor]] = None
    severity_score: Optional[float] = None
    urgency_level: Optional[str] = None  # "Low", "Medium", "High", "Emergency"
    treatment_info: Optional[TreatmentInfo] = None

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    input_symptoms: List[str]
    total_symptoms: int
    analysis_summary: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="Medical Knowledge RAG Chatbot",
    description="A healthcare-focused Retrieval-Augmented Generation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event removed - now handled by lifespan

# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login endpoint for user authentication"""
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user"""
    try:
        new_user = create_user(db, user_data)
        logger.info(f"New user registered: {user_data.username}")
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            is_admin=new_user.is_admin,
            created_at=new_user.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during registration"
        )

# User management endpoints
@app.get("/users", response_model=List[UserResponse])
async def get_users(
    current_user: UserResponse = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    from ..database import User
    users = db.query(User).all()
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
        for user in users
    ]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user by ID"""
    from ..database import User
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Users can only access their own data unless they're admin
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at
    )

# User prediction history endpoint removed

# Landing page - redirect to docs
@app.get("/")
async def root():
    """Landing page - redirects to API documentation"""
    from fastapi.responses import HTMLResponse
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Knowledge RAG Chatbot</title>
        <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
        <style>
            body { margin: 0; padding: 0; }
            .swagger-ui .topbar { display: none; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            const ui = SwaggerUIBundle({
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                layout: 'BaseLayout',
                deepLinking: true,
                showExtensions: true,
                showCommonExtensions: true,
                oauth2RedirectUrl: window.location.origin + '/docs/oauth2-redirect',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ]
            })
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Load disease dataset
def load_disease_dataset():
    """Load the Kaggle disease dataset from CSV files"""
    try:
        # Load main dataset
        dataset_path = Path("data/raw/Disease Dataset/dataset.csv")
        descriptions_path = Path("data/raw/Disease Dataset/symptom_Description.csv")
        precautions_path = Path("data/raw/Disease Dataset/symptom_precaution.csv")
        
        if dataset_path.exists():
            # Load main dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded main dataset with {len(df)} records")
            
            # Load descriptions
            descriptions_df = None
            if descriptions_path.exists():
                descriptions_df = pd.read_csv(descriptions_path)
                logger.info(f"Loaded descriptions for {len(descriptions_df)} diseases")
            
            # Load precautions
            precautions_df = None
            if precautions_path.exists():
                precautions_df = pd.read_csv(precautions_path)
                logger.info(f"Loaded precautions for {len(precautions_df)} diseases")
            
            # Merge descriptions and precautions
            if descriptions_df is not None:
                df = df.merge(descriptions_df, on='Disease', how='left')
            
            if precautions_df is not None:
                df = df.merge(precautions_df, on='Disease', how='left')
            
            logger.info(f"Final dataset with {len(df)} records and {len(df.columns)} columns")
            return df
        else:
            logger.warning("Kaggle dataset not found, using mock data")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

# Global variables to store datasets
disease_dataset = None
drug_database = None
symptom_descriptions = None
symptom_precautions = None

# Load symptom descriptions
def load_symptom_descriptions():
    """Load symptom descriptions from Kaggle dataset"""
    global symptom_descriptions
    try:
        desc_path = Path("data/raw/Disease Dataset/symptom_Description.csv")
        if desc_path.exists():
            symptom_descriptions = pd.read_csv(desc_path)
            logger.info(f"Loaded symptom descriptions with {len(symptom_descriptions)} diseases")
        else:
            logger.warning("Symptom descriptions not found")
            symptom_descriptions = None
    except Exception as e:
        logger.error(f"Error loading symptom descriptions: {e}")
        symptom_descriptions = None
    return symptom_descriptions

# Load symptom precautions
def load_symptom_precautions():
    """Load symptom precautions from Kaggle dataset"""
    global symptom_precautions
    try:
        prec_path = Path("data/raw/Disease Dataset/symptom_precaution.csv")
        if prec_path.exists():
            symptom_precautions = pd.read_csv(prec_path)
            logger.info(f"Loaded symptom precautions with {len(symptom_precautions)} diseases")
        else:
            logger.warning("Symptom precautions not found")
            symptom_precautions = None
    except Exception as e:
        logger.error(f"Error loading symptom precautions: {e}")
        symptom_precautions = None
    return symptom_precautions

# Load drug database
def load_drug_database():
    """Load drug information database"""
    global drug_database
    try:
        drug_path = Path("data/raw/drug_database.json")
        if drug_path.exists():
            with open(drug_path, 'r') as f:
                drug_database = json.load(f)
            logger.info(f"Loaded drug database with {len(drug_database)} diseases")
        else:
            logger.warning("Drug database not found")
            drug_database = {}
    except Exception as e:
        logger.error(f"Error loading drug database: {e}")
        drug_database = {}

# Get treatment information for a disease
def get_treatment_info(disease_name: str) -> Optional[TreatmentInfo]:
    """Get treatment information for a disease"""
    global drug_database
    
    if drug_database is None:
        load_drug_database()
    
    if disease_name in drug_database:
        disease_info = drug_database[disease_name]
        
        # Convert drug names to DrugInfo objects
        drugs = []
        for drug_name in disease_info.get("drugs", []):
            # Determine drug type based on name
            drug_type = "Prescription"
            if any(keyword in drug_name.lower() for keyword in ["vitamin", "supplement", "zinc"]):
                drug_type = "Supplement"
            elif any(keyword in drug_name.lower() for keyword in ["paracetamol", "ibuprofen", "antacids"]):
                drug_type = "Over-the-counter"
            
            drugs.append(DrugInfo(name=drug_name, type=drug_type))
        
        return TreatmentInfo(
            drugs=drugs,
            treatments=disease_info.get("treatments", []),
            duration=disease_info.get("duration", "Varies"),
            notes=f"Consult healthcare provider for proper diagnosis and treatment"
        )
    
    return None

# Risk factor analysis function
def analyze_risk_factors(disease_name: str, symptoms: List[str], age: Optional[int] = None, gender: Optional[str] = None) -> List[RiskFactor]:
    """Analyze risk factors based on disease, symptoms, age, and gender"""
    risk_factors = []
    
    # Age-based risk factors
    if age is not None:
        if age < 5:
            risk_factors.append(RiskFactor(
                factor="Young Age",
                level="High",
                description="Children under 5 are more susceptible to infections and may require immediate medical attention"
            ))
        elif age > 65:
            risk_factors.append(RiskFactor(
                factor="Advanced Age",
                level="High",
                description="Adults over 65 have higher risk of complications and may need specialized care"
            ))
        elif age > 50:
            risk_factors.append(RiskFactor(
                factor="Middle Age",
                level="Medium",
                description="Increased risk of chronic conditions and complications"
            ))
    
    # Gender-based risk factors
    if gender is not None:
        if gender.lower() in ['female', 'f']:
            risk_factors.append(RiskFactor(
                factor="Gender",
                level="Low",
                description="Some conditions may present differently in females"
            ))
        elif gender.lower() in ['male', 'm']:
            risk_factors.append(RiskFactor(
                factor="Gender",
                level="Low",
                description="Some conditions may present differently in males"
            ))
    
    # Symptom-based risk factors
    high_risk_symptoms = ['chest pain', 'shortness of breath', 'severe headache', 'loss of consciousness', 'severe abdominal pain']
    medium_risk_symptoms = ['fever', 'dizziness', 'nausea', 'vomiting', 'diarrhea']
    
    for symptom in symptoms:
        if symptom.lower() in high_risk_symptoms:
            risk_factors.append(RiskFactor(
                factor=f"Symptom: {symptom.title()}",
                level="High",
                description="This symptom may indicate a serious condition requiring immediate attention"
            ))
        elif symptom.lower() in medium_risk_symptoms:
            risk_factors.append(RiskFactor(
                factor=f"Symptom: {symptom.title()}",
                level="Medium",
                description="This symptom should be monitored and may require medical evaluation"
            ))
    
    # Disease-specific risk factors
    if disease_name.lower() in ['malaria', 'dengue', 'typhoid']:
        risk_factors.append(RiskFactor(
            factor="Infectious Disease",
            level="High",
            description="Infectious diseases require immediate medical attention and may need isolation"
        ))
    elif disease_name.lower() in ['diabetes', 'hypertension', 'heart disease']:
        risk_factors.append(RiskFactor(
            factor="Chronic Condition",
            level="High",
            description="Chronic conditions require ongoing medical management and monitoring"
        ))
    
    return risk_factors

# Severity and urgency analysis
def calculate_severity_score(disease_name: str, symptoms: List[str], confidence: float) -> tuple[float, str]:
    """Calculate severity score and urgency level"""
    base_score = confidence * 10  # Base score from 0-10
    
    # Adjust based on disease severity
    high_severity_diseases = ['malaria', 'dengue', 'typhoid', 'pneumonia', 'meningitis']
    medium_severity_diseases = ['diabetes', 'hypertension', 'asthma', 'migraine']
    
    if disease_name.lower() in high_severity_diseases:
        base_score += 3
    elif disease_name.lower() in medium_severity_diseases:
        base_score += 1
    
    # Adjust based on symptom severity
    severe_symptoms = ['chest pain', 'shortness of breath', 'severe headache', 'loss of consciousness']
    moderate_symptoms = ['fever', 'dizziness', 'nausea', 'vomiting']
    
    for symptom in symptoms:
        if symptom.lower() in severe_symptoms:
            base_score += 2
        elif symptom.lower() in moderate_symptoms:
            base_score += 1
    
    # Cap the score at 10
    severity_score = min(base_score, 10)
    
    # Determine urgency level
    if severity_score >= 8:
        urgency_level = "Emergency"
    elif severity_score >= 6:
        urgency_level = "High"
    elif severity_score >= 4:
        urgency_level = "Medium"
    else:
        urgency_level = "Low"
    
    return severity_score, urgency_level

# LLM-style precaution generation
def generate_llm_precautions(disease_name: str, symptoms: List[str], severity_score: float, urgency_level: str) -> List[str]:
    """Generate comprehensive precautions using LLM-style logic"""
    precautions = []
    
    # Base precautions for all conditions
    precautions.extend([
        "Monitor symptoms closely",
        "Stay hydrated and get adequate rest",
        "Avoid self-medication without medical advice"
    ])
    
    # Urgency-based precautions
    if urgency_level == "Emergency":
        precautions.extend([
            "Seek immediate medical attention",
            "Call emergency services if symptoms worsen",
            "Do not delay medical care"
        ])
    elif urgency_level == "High":
        precautions.extend([
            "Schedule urgent medical appointment",
            "Monitor vital signs regularly",
            "Have emergency contacts ready"
        ])
    elif urgency_level == "Medium":
        precautions.extend([
            "Schedule medical appointment within 24-48 hours",
            "Keep symptom diary",
            "Avoid strenuous activities"
        ])
    else:  # Low urgency
        precautions.extend([
            "Monitor symptoms for 24-48 hours",
            "Schedule routine medical checkup if symptoms persist",
            "Maintain healthy lifestyle"
        ])
    
    # Disease-specific precautions
    if disease_name.lower() in ['malaria', 'dengue', 'typhoid']:
        precautions.extend([
            "Isolate to prevent transmission",
            "Use mosquito nets and repellents",
            "Avoid sharing personal items"
        ])
    elif disease_name.lower() in ['diabetes', 'hypertension']:
        precautions.extend([
            "Monitor blood sugar/blood pressure regularly",
            "Follow prescribed medication regimen",
            "Maintain healthy diet and exercise"
        ])
    elif disease_name.lower() in ['asthma', 'allergy']:
        precautions.extend([
            "Avoid known triggers and allergens",
            "Keep rescue inhaler/medication handy",
            "Ensure good air quality in living space"
        ])
    
    # Symptom-specific precautions
    if 'fever' in [s.lower() for s in symptoms]:
        precautions.extend([
            "Monitor temperature regularly",
            "Use fever-reducing measures (cool compresses, adequate fluids)",
            "Seek medical attention if fever persists or exceeds 103°F"
        ])
    
    if 'chest pain' in [s.lower() for s in symptoms]:
        precautions.extend([
            "Seek immediate medical attention",
            "Avoid physical exertion",
            "Monitor for signs of heart attack"
        ])
    
    if 'shortness of breath' in [s.lower() for s in symptoms]:
        precautions.extend([
            "Sit upright and try to remain calm",
            "Seek immediate medical attention if breathing becomes difficult",
            "Avoid triggers like smoke or allergens"
        ])
    
    return precautions[:8]  # Limit to 8 most important precautions

# Disease prediction endpoint (protected)
@app.post("/predict-disease", response_model=PredictionResponse)
async def predict_disease(
    symptom_input: SymptomInput, 
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict disease based on symptoms using the actual dataset
    
    Input: List of symptoms
    Output: Predicted diseases with confidence scores
    """
    global disease_dataset
    
    try:
        logger.info(f"Predicting disease for symptoms: {symptom_input.symptoms}")
        
        # Load dataset if not already loaded
        if disease_dataset is None:
            disease_dataset = load_disease_dataset()
        
        predictions = []
        
        if disease_dataset is not None:
            # Use Kaggle dataset for predictions
            input_symptoms = [s.lower().replace(' ', '_') for s in symptom_input.symptoms]
            
            # Group by disease to get unique diseases
            disease_groups = disease_dataset.groupby('Disease')
            
            for disease_name, group in disease_groups:
                # Get all symptoms for this disease across all records
                disease_symptoms = set()
                for _, row in group.iterrows():
                    for col in [f'Symptom_{i}' for i in range(1, 18)]:  # Symptom_1 to Symptom_17
                        if pd.notna(row[col]) and row[col].strip():
                            disease_symptoms.add(row[col].lower().strip())
                
                # Calculate match score
                matches = sum(1 for symptom in input_symptoms if symptom in disease_symptoms)
                if matches > 0:
                    confidence = matches / len(input_symptoms)
                    
                    # Get description and precautions from first row of the group
                    first_row = group.iloc[0]
                    description = first_row.get('Description', '') if pd.notna(first_row.get('Description', '')) else ""
                    
                    # Parse precautions
                    precautions = []
                    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                        if pd.notna(first_row.get(col, '')) and first_row.get(col, '').strip():
                            precautions.append(first_row[col].strip())
                    
                    # Calculate severity and urgency
                    severity_score, urgency_level = calculate_severity_score(disease_name, input_symptoms, confidence)
                    
                    # Analyze risk factors
                    risk_factors = analyze_risk_factors(disease_name, input_symptoms, symptom_input.age, symptom_input.gender)
                    
                    # Generate LLM-style precautions
                    llm_precautions = generate_llm_precautions(disease_name, input_symptoms, severity_score, urgency_level)
                    
                    # Get treatment information
                    treatment_info = get_treatment_info(disease_name)
                    
                    prediction = DiseasePrediction(
                        disease=disease_name,
                        confidence=round(confidence, 2),
                        description=description,
                        precautions=llm_precautions,
                        risk_factors=risk_factors,
                        severity_score=round(severity_score, 1),
                        urgency_level=urgency_level,
                        treatment_info=treatment_info
                    )
                    predictions.append(prediction)
            
            # Sort by confidence and take top 3
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            predictions = predictions[:3]
        
        else:
            # Fallback to mock predictions if dataset not available
            logger.warning("Using mock predictions - dataset not available")
            # Get treatment info for mock predictions
            common_cold_treatment = get_treatment_info("Common Cold")
            influenza_treatment = get_treatment_info("Influenza")
            
            predictions = [
                DiseasePrediction(
                    disease="Common Cold",
                    confidence=0.75,
                    description="Viral infection affecting the upper respiratory tract",
                    precautions=["Rest", "Stay hydrated", "Use humidifier", "Avoid close contact"],
                    treatment_info=common_cold_treatment
                ),
                DiseasePrediction(
                    disease="Influenza",
                    confidence=0.65,
                    description="Viral infection with more severe symptoms than common cold",
                    precautions=["Rest", "Stay hydrated", "Antiviral medication if prescribed", "Isolate from others"],
                    treatment_info=influenza_treatment
                )
            ]
        
        # Generate analysis summary
        if predictions:
            top_prediction = predictions[0]
            analysis_summary = f"Based on your symptoms, the most likely condition is {top_prediction.disease} with {top_prediction.confidence*100:.1f}% confidence. "
            analysis_summary += f"Severity score: {top_prediction.severity_score}/10. "
            analysis_summary += f"Urgency level: {top_prediction.urgency_level}. "
            if top_prediction.urgency_level in ["Emergency", "High"]:
                analysis_summary += "Immediate medical attention is recommended."
            elif top_prediction.urgency_level == "Medium":
                analysis_summary += "Medical evaluation within 24-48 hours is advised."
            else:
                analysis_summary += "Monitor symptoms and consult healthcare provider if they persist."
        else:
            analysis_summary = "No specific disease pattern identified. Please consult a healthcare provider for proper diagnosis."
        
        response = PredictionResponse(
            predictions=predictions,
            input_symptoms=symptom_input.symptoms,
            total_symptoms=len(symptom_input.symptoms),
            analysis_summary=analysis_summary
        )
        
        # Save prediction history to database
        if predictions:
            top_prediction = predictions[0]
            history_entry = PredictionHistory(
                user_id=current_user.id,
                symptoms=json.dumps(symptom_input.symptoms),
                predicted_disease=top_prediction.disease,
                confidence=str(top_prediction.confidence),
                severity_score=str(top_prediction.severity_score),
                urgency_level=top_prediction.urgency_level
            )
            db.add(history_entry)
            db.commit()
            logger.info(f"Saved prediction history for user {current_user.username}")
        
        logger.info(f"Generated {len(predictions)} predictions")
        return response
        
    except Exception as e:
        logger.error(f"Error in disease prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Get available symptoms (for frontend dropdown) - protected
@app.get("/symptoms")
async def get_available_symptoms(current_user: UserResponse = Depends(get_current_user)):
    """Get list of available symptoms from the dataset"""
    global disease_dataset
    
    try:
        # Load dataset if not already loaded
        if disease_dataset is None:
            disease_dataset = load_disease_dataset()
        
        symptoms = set()
        
        if disease_dataset is not None:
            # Extract all symptoms from the Kaggle dataset
            for _, row in disease_dataset.iterrows():
                for col in [f'Symptom_{i}' for i in range(1, 18)]:  # Symptom_1 to Symptom_17
                    if pd.notna(row[col]) and row[col].strip():
                        symptoms.add(row[col].lower().strip())
        else:
            # Fallback to mock symptoms if dataset not available
            symptoms = {
                "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
                "diarrhea", "abdominal pain", "chest pain", "shortness of breath",
                "dizziness", "muscle aches", "sore throat", "runny nose",
                "sneezing", "itchy eyes", "rash", "swelling", "joint pain",
                "back pain", "insomnia", "anxiety", "depression", "weight loss",
                "weight gain", "loss of appetite", "excessive thirst", "frequent urination"
            }
        
        return {"symptoms": sorted(list(symptoms))}
        
    except Exception as e:
        logger.error(f"Error getting symptoms: {e}")
        # Return basic symptoms as fallback
        return {"symptoms": ["fever", "cough", "headache", "fatigue", "nausea"]}

# Get disease information
@app.get("/disease/{disease_name}")
async def get_disease_info(disease_name: str):
    """Get detailed information about a specific disease"""
    global disease_dataset, symptom_descriptions, symptom_precautions
    
    try:
        # Load datasets if not already loaded
        if disease_dataset is None:
            disease_dataset = load_disease_dataset()
        if symptom_descriptions is None:
            symptom_descriptions = load_symptom_descriptions()
        if symptom_precautions is None:
            symptom_precautions = load_symptom_precautions()
        
        # Search for disease in dataset (case-insensitive)
        disease_found = None
        disease_lower = disease_name.lower()
        
        if disease_dataset is not None:
            for _, row in disease_dataset.iterrows():
                if disease_lower in row['Disease'].lower():
                    disease_found = row
                    break
        
        if disease_found is not None:
            disease_name_found = disease_found['Disease']
            
            # Get symptoms for this disease
            symptoms = []
            for i in range(1, 18):  # Symptom_1 to Symptom_17
                col_name = f'Symptom_{i}'
                if col_name in disease_found and pd.notna(disease_found[col_name]) and disease_found[col_name].strip():
                    symptoms.append(disease_found[col_name].strip())
            
            # Get description
            description = "No description available"
            if symptom_descriptions is not None:
                desc_row = symptom_descriptions[symptom_descriptions['Disease'] == disease_name_found]
                if not desc_row.empty:
                    description = desc_row.iloc[0]['Description']
            
            # Get precautions
            precautions = []
            if symptom_precautions is not None:
                prec_row = symptom_precautions[symptom_precautions['Disease'] == disease_name_found]
                if not prec_row.empty:
                    for i in range(1, 5):  # Precaution_1 to Precaution_4
                        col_name = f'Precaution_{i}'
                        if col_name in prec_row.columns and pd.notna(prec_row.iloc[0][col_name]) and prec_row.iloc[0][col_name].strip():
                            precautions.append(prec_row.iloc[0][col_name].strip())
            
            # Get treatment information
            treatment_info = get_treatment_info(disease_name_found)
            
            return {
                "disease": disease_name_found,
                "description": description,
                "symptoms": symptoms,
                "precautions": precautions,
                "treatment_info": treatment_info.dict() if treatment_info else None
            }
        else:
            # Fallback to mock data for common diseases
            mock_diseases = {
                "common cold": {
                    "disease": "Common Cold",
                    "description": "Viral infection of the upper respiratory tract",
                    "symptoms": ["runny nose", "sneezing", "cough", "sore throat"],
                    "precautions": ["Rest", "Stay hydrated", "Use humidifier"],
                    "treatment_info": get_treatment_info("Common Cold").dict() if get_treatment_info("Common Cold") else None
                },
                "influenza": {
                    "disease": "Influenza",
                    "description": "Viral infection with more severe symptoms",
                    "symptoms": ["fever", "chills", "muscle aches", "fatigue", "cough"],
                    "precautions": ["Rest", "Stay hydrated", "Antiviral medication"],
                    "treatment_info": get_treatment_info("Influenza").dict() if get_treatment_info("Influenza") else None
                }
            }
            
            disease_key = disease_name.lower().replace(" ", "_")
            if disease_key in mock_diseases:
                return mock_diseases[disease_key]
            else:
                raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found in database")
                
    except Exception as e:
        logger.error(f"Error getting disease info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving disease information: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )