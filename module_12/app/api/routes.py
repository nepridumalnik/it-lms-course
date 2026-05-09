from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.utils.model_service import ModelNotReadyError, model_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Вернуть статус сервиса."""
    return HealthResponse(status="ok")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={503: {"model": ErrorResponse}},
)
def model_info() -> ModelInfoResponse:
    """Вернуть параметры и качество модели."""
    try:
        return model_service.get_model_info()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Оценить кредитную заявку."""
    try:
        return model_service.predict(payload)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
