from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = ARTIFACTS_DIR / "credit_risk_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "credit_risk_metadata.joblib"
TRAINING_DATA_PATH = DATA_DIR / "sample_customers.csv"
