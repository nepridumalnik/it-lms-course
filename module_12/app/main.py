from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request

from app.api.routes import router
from app.core.logging import configure_logging
from app.utils.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    model_service.load_or_train()
    yield


app = FastAPI(
    title="ML-сервис кредитного скоринга",
    description="REST-сервис для оценки вероятности одобрения кредита на scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logging.info(
        "Запрос %s %s -> %s %.2f мс",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(router)
