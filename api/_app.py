"""
Full MediConnect backend — imported by index.py.
Kept in a separate file so Vercel's module scanner does not
iterate over SQLAlchemy model classes (which break issubclass checks).
"""
import os
import json
from datetime import datetime, timedelta
from typing import Optional, List
from functools import lru_cache

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

import google.generativeai as genai

# ─── Config ───────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/mediconnect"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    GEMINI_API_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# ─── Database ─────────────────────────────────────────────────────────────────

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
DBBase = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Models ───────────────────────────────────────────────────────────────────

class User(DBBase):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    medical_profile = relationship("MedicalProfile", back_populates="user", uselist=False)
    health_logs = relationship("HealthLog", back_populates="user", order_by="HealthLog.created_at.desc()")

class MedicalProfile(DBBase):
    __tablename__ = "medical_profile"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    age = Column(Integer)
    primary_condition = Column(String(255), default="")
    current_treatments = Column(String(500), default="")
    allergies = Column(String(500), default="")
    user = relationship("User", back_populates="medical_profile")

class HealthLog(DBBase):
    __tablename__ = "health_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    module_used = Column(String(50), nullable=False)
    user_input = Column(Text, nullable=False)
    ai_response = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="health_logs")

DBBase.metadata.create_all(bind=engine)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MedicalProfileCreate(BaseModel):
    age: Optional[int] = None
    primary_condition: Optional[str] = ""
    current_treatments: Optional[str] = ""
    allergies: Optional[str] = ""

class MedicalProfileOut(BaseModel):
    id: int
    user_id: int
    age: Optional[int]
    primary_condition: Optional[str]
    current_treatments: Optional[str]
    allergies: Optional[str]
    class Config:
        from_attributes = True

class SymptomTriageInput(BaseModel):
    symptoms: str

class ReportSimplifyInput(BaseModel):
    report_text: str

class DietGeneratorInput(BaseModel):
    condition: str

class DrugInteractionInput(BaseModel):
    medications: List[str]

class OTCFirstAidInput(BaseModel):
    symptoms: str

class HealthLogOut(BaseModel):
    id: int
    module_used: str
    user_input: str
    ai_response: Optional[dict]
    created_at: datetime
    class Config:
        from_attributes = True

# ─── Auth ─────────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> "User":
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

# ─── AI Service ───────────────────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "⚠️ DISCLAIMER: I am an AI, not a doctor. "
    "Please consult your oncologist or physician before making medical decisions."
)

SYSTEM_PROMPTS = {
    "SYMPTOM_TRIAGE": (
        "You are a medical triage AI. The user will provide symptoms. "
        "Identify potential causes, focusing on general health but keeping high awareness "
        "for oncological red flags (e.g., unexplained weight loss, persistent lumps). "
        "Output the recommended specialist (e.g., Medical Oncologist, General Physician) "
        "and an urgency level (Low, Medium, High). "
        'Return as JSON with keys: "potential_causes" (list of strings), '
        '"recommended_specialist" (string), "urgency" (string: Low/Medium/High), '
        '"explanation" (string).'
    ),
    "REPORT_SIMPLIFIER": (
        "You are an empathetic medical translator. The user will paste text from a lab or "
        "biopsy report (e.g., CBC, tumor markers like CEA/CA-125, or histology reports). "
        "Explain the terms in 'Explain Like I'm 5' plain English. Do not diagnose. "
        "Explain what the markers generally mean and if the values are typically considered "
        "high or low based on standard reference ranges. "
        'Return as JSON with keys: "simplified_explanation" (string), '
        '"key_terms" (list of objects with "term" and "meaning"), '
        '"general_assessment" (string).'
    ),
    "DIET_GENERATOR": (
        "You are an oncology and clinical nutritionist. The user will provide their condition "
        "(e.g., undergoing chemotherapy, post-surgery, or general illness like Typhoid). "
        "Generate a 3-day meal plan. If the user is a cancer patient, focus on foods that "
        "combat nausea, maintain weight, and are easy to swallow/digest. "
        "Return strictly in JSON format: "
        '{"day_1": {"breakfast": "", "lunch": "", "dinner": "", "snack": ""}, '
        '"day_2": {"breakfast": "", "lunch": "", "dinner": "", "snack": ""}, '
        '"day_3": {"breakfast": "", "lunch": "", "dinner": "", "snack": ""}, '
        '"tips": ["tip1", "tip2"]}.'
    ),
    "DRUG_INTERACTION": (
        "You are a clinical pharmacist AI. The user will provide a list of medications. "
        "Check for known interactions, especially focusing on how common OTC drugs "
        "(like Ibuprofen, Omeprazole, or herbal supplements) interact with targeted therapies "
        "or chemotherapy drugs (e.g., Tamoxifen, Methotrexate). "
        "Return a risk level (Safe, Moderate, High) and a brief 2-sentence explanation. "
        'Return as JSON with keys: "interactions" (list of objects with "drug_pair", '
        '"risk_level" (Safe/Moderate/High), "explanation"), '
        '"overall_risk" (Safe/Moderate/High), "summary" (string).'
    ),
    "OTC_FIRST_AID": (
        "Suggest standard OTC medications or home remedies for minor symptoms "
        "(e.g., mild fever, chemotherapy-induced mild nausea, headache). "
        "Always add a disclaimer that cancer patients must clear any OTC drugs with their "
        "oncologist first to avoid liver/kidney strain. "
        'Return as JSON with keys: "recommendations" (list of objects with "remedy" and '
        '"usage_notes"), "warnings" (list of strings), "cancer_patient_note" (string).'
    ),
}

def _configure_genai():
    genai.configure(api_key=settings.GEMINI_API_KEY)

def call_ai(prompt_key: str, user_message: str) -> dict:
    _configure_genai()
    system_prompt = SYSTEM_PROMPTS.get(prompt_key, "")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    try:
        response = model.generate_content(user_message)
        raw_text = response.text.strip()
        parsed = json.loads(raw_text)
        parsed["disclaimer"] = MEDICAL_DISCLAIMER
        return parsed
    except json.JSONDecodeError:
        return {
            "raw_response": response.text if response else "",
            "disclaimer": MEDICAL_DISCLAIMER,
            "error": "AI returned non-JSON response. Showing raw text.",
        }
    except Exception as e:
        return {
            "error": f"AI service unavailable: {str(e)}",
            "disclaimer": MEDICAL_DISCLAIMER,
        }

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MediConnect API",
    description="AI-powered Smart Healthcare Assistant with Oncology Focus",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth Routes ──────────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    profile = MedicalProfile(user_id=user.id)
    db.add(profile)
    db.commit()
    return user

@auth_router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

@auth_router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ─── Profile Routes ──────────────────────────────────────────────────────────

profile_router = APIRouter(prefix="/api/v1/profile", tags=["Medical Profile"])

@profile_router.get("/", response_model=MedicalProfileOut)
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(MedicalProfile).filter(MedicalProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@profile_router.put("/", response_model=MedicalProfileOut)
def update_profile(
    payload: MedicalProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.query(MedicalProfile).filter(MedicalProfile.user_id == current_user.id).first()
    if not profile:
        profile = MedicalProfile(user_id=current_user.id)
        db.add(profile)
    if payload.age is not None:
        profile.age = payload.age
    if payload.primary_condition is not None:
        profile.primary_condition = payload.primary_condition
    if payload.current_treatments is not None:
        profile.current_treatments = payload.current_treatments
    if payload.allergies is not None:
        profile.allergies = payload.allergies
    db.commit()
    db.refresh(profile)
    return profile

# ─── AI Module Routes ────────────────────────────────────────────────────────

ai_router = APIRouter(prefix="/api/v1", tags=["AI Modules"])

def _log_interaction(db: Session, user_id: int, module: str, user_input: str, ai_resp: dict):
    log = HealthLog(
        user_id=user_id,
        module_used=module,
        user_input=user_input,
        ai_response=ai_resp,
    )
    db.add(log)
    db.commit()

@ai_router.post("/symptom-triage")
def symptom_triage(payload: SymptomTriageInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = call_ai("SYMPTOM_TRIAGE", payload.symptoms)
    _log_interaction(db, current_user.id, "SYMPTOM_TRIAGE", payload.symptoms, result)
    return result

@ai_router.post("/simplify-report")
def simplify_report(payload: ReportSimplifyInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = call_ai("REPORT_SIMPLIFIER", payload.report_text)
    _log_interaction(db, current_user.id, "REPORT_SIMPLIFIER", payload.report_text, result)
    return result

@ai_router.post("/diet-generator")
def diet_generator(payload: DietGeneratorInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = call_ai("DIET_GENERATOR", payload.condition)
    _log_interaction(db, current_user.id, "DIET_GENERATOR", payload.condition, result)
    return result

@ai_router.post("/drug-interaction")
def drug_interaction(payload: DrugInteractionInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    meds_str = ", ".join(payload.medications)
    result = call_ai("DRUG_INTERACTION", meds_str)
    _log_interaction(db, current_user.id, "DRUG_INTERACTION", meds_str, result)
    return result

@ai_router.post("/otc-first-aid")
def otc_first_aid(payload: OTCFirstAidInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = call_ai("OTC_FIRST_AID", payload.symptoms)
    _log_interaction(db, current_user.id, "OTC_FIRST_AID", payload.symptoms, result)
    return result

# ─── History Routes ──────────────────────────────────────────────────────────

history_router = APIRouter(prefix="/api/v1/history", tags=["Health Logs"])

@history_router.get("/", response_model=List[HealthLogOut])
def get_history(
    module: Optional[str] = Query(None, description="Filter by module name"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(HealthLog).filter(HealthLog.user_id == current_user.id)
    if module:
        query = query.filter(HealthLog.module_used == module)
    return query.order_by(HealthLog.created_at.desc()).limit(limit).all()

# ─── Register Routers ────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(ai_router)
app.include_router(history_router)

@app.get("/")
def root():
    return {"message": "MediConnect API is running", "docs": "/docs"}
