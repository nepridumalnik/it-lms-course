import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import ARTIFACTS_DIR, DATA_DIR, METADATA_PATH, MODEL_PATH, TRAINING_DATA_PATH
from app.models.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_debt",
    "debt_to_income",
    "loan_to_income",
]

DATASET_TYPE = "synthetic_credit_scoring"
DECISION_THRESHOLD = 0.5
MODEL_CLASS_MEANING = "internal model predicts default_risk; API class is approval decision"


class ModelNotReadyError(RuntimeError):
    pass


@dataclass
class ModelMetadata:
    feature_names: List[str]
    accuracy: float
    roc_auc: float
    threshold: float
    best_params: Dict[str, Any]
    dataset_type: str


class ModelService:
    """Сервис загрузки модели, обучения и скоринга заявок."""

    def __init__(self) -> None:
        self.model: Pipeline | None = None
        self.metadata: ModelMetadata | None = None

    def load_or_train(self) -> None:
        """Загрузить готовую модель или обучить новую."""
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        if MODEL_PATH.exists() and METADATA_PATH.exists():
            loaded_model = joblib.load(MODEL_PATH)
            loaded_metadata = joblib.load(METADATA_PATH)
            if self._metadata_is_compatible(loaded_metadata):
                self.model = loaded_model
                self.metadata = loaded_metadata
                LOGGER.info("Модель загружена из %s", MODEL_PATH)
                return

            LOGGER.info("Найдены старые метаданные модели. Запускаю переобучение.")

        LOGGER.info("Обучаю модель с подбором гиперпараметров.")
        X, y = self._build_dataset(TRAINING_DATA_PATH)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
            ]
        )
        param_grid = {
            "classifier__n_estimators": [80, 140],
            "classifier__max_depth": [5, None],
            "classifier__min_samples_leaf": [2, 5],
            "classifier__class_weight": [None, "balanced"],
        }
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        probabilities = best_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
        metadata = ModelMetadata(
            feature_names=FEATURE_NAMES,
            accuracy=float(accuracy_score(y_test, predictions)),
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            threshold=DECISION_THRESHOLD,
            best_params=dict(search.best_params_),
            dataset_type=DATASET_TYPE,
        )

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(metadata, METADATA_PATH)
        self.model = best_model
        self.metadata = metadata
        LOGGER.info(
            "Модель обучена: accuracy=%.3f roc_auc=%.3f best_params=%s",
            metadata.accuracy,
            metadata.roc_auc,
            metadata.best_params,
        )

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        """Вернуть бизнес-решение и вероятности по одной заявке."""
        if self.model is None or self.metadata is None:
            raise ModelNotReadyError("Модель не загружена.")

        features = self._features_from_payload(payload)
        X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
        risk_probability = float(self.model.predict_proba(X)[0, 1])
        approved = risk_probability < self.metadata.threshold
        approval_probability = 1 - risk_probability

        return PredictionResponse(
            class_=1 if approved else 0,
            decision="одобрить" if approved else "отказать",
            model_target="default_risk",
            model_class_meaning=MODEL_CLASS_MEANING,
            probability_approved=round(approval_probability, 4),
            probability_risk=round(risk_probability, 4),
            decision_threshold=self.metadata.threshold,
            features={key: round(value, 4) for key, value in features.items()},
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Вернуть метаданные текущей модели."""
        if self.metadata is None:
            raise ModelNotReadyError("Модель не загружена.")

        return ModelInfoResponse(
            feature_names=self.metadata.feature_names,
            accuracy=round(self.metadata.accuracy, 4),
            roc_auc=round(self.metadata.roc_auc, 4),
            threshold=self.metadata.threshold,
            best_params=self.metadata.best_params,
            dataset_type=self.metadata.dataset_type,
        )

    def _metadata_is_compatible(self, metadata: object) -> bool:
        """Проверить, что артефакт создан новой версией обучения."""
        required_fields = [
            "feature_names",
            "accuracy",
            "roc_auc",
            "threshold",
            "best_params",
            "dataset_type",
        ]
        return all(hasattr(metadata, field) for field in required_fields)

    def _features_from_payload(self, payload: PredictionRequest) -> Dict[str, float]:
        """Собрать признаки из входного запроса."""
        annual_income = float(payload.annual_income)
        if annual_income <= 0:
            raise ValueError("Поле annual_income должно быть больше 0 для расчета долей.")

        return {
            "age": float(payload.age),
            "annual_income": annual_income,
            "loan_amount": float(payload.loan_amount),
            "credit_score": float(payload.credit_score),
            "employment_years": float(payload.employment_years),
            "existing_debt": float(payload.existing_debt),
            "debt_to_income": float(payload.existing_debt) / annual_income,
            "loan_to_income": float(payload.loan_amount) / annual_income,
        }

    def _build_dataset(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Прочитать датасет и подготовить матрицу признаков."""
        if not path.exists():
            self._generate_training_data(path)

        rows: List[List[float]] = []
        labels: List[int] = []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                annual_income = float(row["annual_income"])
                existing_debt = float(row["existing_debt"])
                loan_amount = float(row["loan_amount"])
                rows.append(
                    [
                        float(row["age"]),
                        annual_income,
                        loan_amount,
                        float(row["credit_score"]),
                        float(row["employment_years"]),
                        existing_debt,
                        existing_debt / annual_income,
                        loan_amount / annual_income,
                    ]
                )
                labels.append(int(row["default_risk"]))

        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)

    def _generate_training_data(self, path: Path) -> None:
        """Создать синтетический датасет для учебного скоринга."""
        rng = np.random.default_rng(42)
        path.parent.mkdir(exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as file:
            fieldnames = [
                "age",
                "annual_income",
                "loan_amount",
                "credit_score",
                "employment_years",
                "existing_debt",
                "default_risk",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for _ in range(700):
                age = int(rng.integers(21, 76))
                annual_income = float(rng.lognormal(mean=11.0, sigma=0.45))
                annual_income = float(np.clip(annual_income, 18_000, 250_000))
                loan_amount = float(rng.uniform(2_000, 120_000))
                credit_score = int(np.clip(rng.normal(675, 85), 300, 850))
                employment_years = float(np.clip(rng.gamma(shape=2.2, scale=3.0), 0, 40))
                existing_debt = float(rng.uniform(0, annual_income * 1.2))

                debt_to_income = existing_debt / annual_income
                loan_to_income = loan_amount / annual_income
                risk_score = (
                    1.8 * debt_to_income
                    + 1.4 * loan_to_income
                    + max(0, 680 - credit_score) / 120
                    - employment_years / 18
                    - max(0, annual_income - 55_000) / 180_000
                )
                default_probability = 1 / (1 + np.exp(-(risk_score - 1.15)))
                default_risk = int(rng.random() < default_probability)

                writer.writerow(
                    {
                        "age": age,
                        "annual_income": round(annual_income, 2),
                        "loan_amount": round(loan_amount, 2),
                        "credit_score": credit_score,
                        "employment_years": round(employment_years, 2),
                        "existing_debt": round(existing_debt, 2),
                        "default_risk": default_risk,
                    }
                )


model_service = ModelService()
