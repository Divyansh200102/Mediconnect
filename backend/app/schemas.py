from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# --- Auth ---
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


# --- Medical Profile ---
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


# --- AI Module Inputs ---
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


# --- Health Log ---
class HealthLogOut(BaseModel):
    id: int
    module_used: str
    user_input: str
    ai_response: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True
