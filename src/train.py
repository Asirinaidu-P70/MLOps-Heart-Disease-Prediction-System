from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "heart.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "Heart Disease Prediction"


def get_target_column(df: pd.DataFrame) -> str:
    for candidate in ["target", "condition", "class", "num", "output"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find target column. Available columns: {list(df.columns)}")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run: python src/download_data.py")

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    df.columns = [str(c).strip().lower() for c in df.columns]
    target_col = get_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    y = y.astype(int)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )),
    ])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        print("MLflow experiment:", MLFLOW_EXPERIMENT_NAME)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "test_size": 0.2,
            "random_state": 42,
            "features": len(feature_names),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(DATA), artifact_path="dataset")
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, MODEL_DIR / "model.joblib")
        (MODEL_DIR / "features.json").write_text(json.dumps(feature_names, indent=2))
        (MODEL_DIR / "model_info.json").write_text(json.dumps({
            "run_id": run.info.run_id,
            "target_column": target_col,
            "metrics": metrics,
            "feature_names": feature_names,
        }, indent=2))

        # Save the reference feature distribution for Evidently monitoring.
        X_train.to_csv(REPORT_DIR / "reference_data.csv", index=False)

        print("Training complete")
        print(json.dumps(metrics, indent=2))
        print("MLflow run:", run.info.run_id)
        print("Model:", MODEL_DIR / "model.joblib")


if __name__ == "__main__":
    main()
