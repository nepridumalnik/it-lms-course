from fastapi import APIRouter, HTTPException

from app.models.schemas import ErrorResponse, HealthResponse, PredictionRequest, PredictionResponse
from app.utils.model_service import ModelNotReadyError, model_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return model_service.predict(payload)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
