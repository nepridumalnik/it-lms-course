from typing import Dict, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years.")
    annual_income: float = Field(..., ge=0, le=1_000_000, description="Annual income in USD.")
    loan_amount: float = Field(..., gt=0, le=500_000, description="Requested loan amount in USD.")
    credit_score: int = Field(..., ge=300, le=850, description="FICO-like credit score.")
    employment_years: float = Field(..., ge=0, le=60, description="Years employed.")
    existing_debt: float = Field(..., ge=0, le=1_000_000, description="Existing total debt in USD.")


class PredictionResponse(BaseModel):
    prediction: Literal["approved", "rejected"]
    risk_probability: float = Field(..., ge=0, le=1)
    approval_probability: float = Field(..., ge=0, le=1)
    decision_threshold: float
    features: Dict[str, float]


class ErrorResponse(BaseModel):
    detail: str
