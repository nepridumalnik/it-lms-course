from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Возраст клиента в годах.")
    annual_income: float = Field(..., ge=0, le=1_000_000, description="Годовой доход клиента.")
    loan_amount: float = Field(..., gt=0, le=500_000, description="Запрошенная сумма кредита.")
    credit_score: int = Field(..., ge=300, le=850, description="Кредитный рейтинг клиента.")
    employment_years: float = Field(..., ge=0, le=60, description="Стаж работы в годах.")
    existing_debt: float = Field(..., ge=0, le=1_000_000, description="Текущая сумма долгов.")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: int = Field(..., ge=0, le=1, alias="class", description="1 - одобрить, 0 - отказать.")
    decision: Literal["одобрить", "отказать"]
    model_target: Literal["default_risk"]
    model_class_meaning: str
    probability_approved: float = Field(..., ge=0, le=1)
    probability_risk: float = Field(..., ge=0, le=1)
    decision_threshold: float
    features: Dict[str, float]


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    accuracy: float
    roc_auc: float
    threshold: float
    best_params: Dict[str, Any]
    dataset_type: str


class ErrorResponse(BaseModel):
    detail: str
