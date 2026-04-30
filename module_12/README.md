# Local ML Scoring Service

FastAPI service for local loan approval scoring. It uses a small scikit-learn model trained on synthetic customer data generated inside `data/`. No paid APIs and no external model downloads are required.

## Architecture

- `app/main.py` - FastAPI app, startup loading, request logging.
- `app/api/routes.py` - `/health` and `/predict` endpoints.
- `app/models/schemas.py` - Pydantic request and response validation.
- `app/utils/model_service.py` - dataset generation, training, artifact loading, prediction.
- `data/` - generated training CSV.
- `artifacts/` - trained model and metadata.

At startup the service loads `artifacts/credit_risk_model.joblib`. If it does not exist, it generates `data/sample_customers.csv`, trains a RandomForest model, and saves artifacts.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{"status":"ok"}
```

## Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "annual_income": 82000,
    "loan_amount": 18000,
    "credit_score": 735,
    "employment_years": 6,
    "existing_debt": 12000
  }'
```

Example response:

```json
{
  "prediction": "approved",
  "risk_probability": 0.1875,
  "approval_probability": 0.8125,
  "decision_threshold": 0.5,
  "features": {
    "age": 34.0,
    "annual_income": 82000.0,
    "loan_amount": 18000.0,
    "credit_score": 735.0,
    "employment_years": 6.0,
    "existing_debt": 12000.0,
    "debt_to_income": 0.1463,
    "loan_to_income": 0.2195
  }
}
```

## Validation

The service validates:

- age from 18 to 100
- annual income from 0 to 1,000,000
- loan amount greater than 0
- credit score from 300 to 850
- employment years from 0 to 60
- existing debt from 0 to 1,000,000

Invalid requests return HTTP `422` with Pydantic details. Runtime model errors return meaningful HTTP errors.

## Limitations

- Dataset is synthetic and for demonstration only.
- Model output is not suitable for real credit decisions.
- No authentication or persistent database.
- Retraining happens only when artifacts are missing.
