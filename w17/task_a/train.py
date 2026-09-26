"""
Train 3 models on Telco Churn dataset with MLflow tracking.

Pipeline:
  load → preprocess → train 3 configs → log params/metrics/artifacts → register best model
"""

import os
import pathlib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = pathlib.Path(__file__).parent / "data" / "telco_churn.csv"
EXPERIMENT_NAME = "telco_churn_experiment"
MODEL_NAME = "ChurnModel"
MLFLOW_URI = "http://127.0.0.1:5000"
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_and_preprocess(path: pathlib.Path):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode all string/object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_confusion_matrix(y_true, y_pred, run_name: str) -> pathlib.Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — {run_name}")
    path = REPORTS_DIR / f"confusion_matrix_{run_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_prob, run_name: str) -> pathlib.Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {run_name}")
    ax.legend()
    path = REPORTS_DIR / f"roc_curve_{run_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


RUNS = [
    {
        "name": "logreg_C0.1",
        "params": {"model_type": "LogisticRegression", "C": 0.1, "max_iter": 500},
        "model": LogisticRegression(C=0.1, max_iter=500, random_state=42),
    },
    {
        "name": "logreg_C10",
        "params": {"model_type": "LogisticRegression", "C": 10, "max_iter": 500},
        "model": LogisticRegression(C=10, max_iter=500, random_state=42),
    },
    {
        "name": "rf_depth5",
        "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 5},
        "model": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    },
    {
        "name": "xgb_default",
        "params": {"model_type": "XGBoost", "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        "model": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               random_state=42, eval_metric="logloss", verbosity=0),
    },
]


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_run_id = None
    best_f1 = -1.0
    best_run_name = None

    for run_cfg in RUNS:
        name = run_cfg["name"]
        model = run_cfg["model"]
        params = run_cfg["params"]

        # Tree models don't need scaling
        needs_scaling = params["model_type"] in ("LogisticRegression",)
        Xtr = X_train_s if needs_scaling else X_train.values
        Xte = X_test_s if needs_scaling else X_test.values

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)
            print(f"[{name}] F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

            # Artifacts
            cm_path = save_confusion_matrix(y_test, y_pred, name)
            roc_path = save_roc_curve(y_test, y_prob, name)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            if params.get("model_type") == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            elif params.get("model_type") == "RandomForest":
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_run_id = run.info.run_id
                best_run_name = name

    # Register best model and transition stages
    print(f"\nBest run: {best_run_name} (F1={best_f1:.4f}), run_id={best_run_id}")
    client = MlflowClient(MLFLOW_URI)

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version

    # Transition: None → Staging → Production
    client.transition_model_version_stage(MODEL_NAME, version, "Staging")
    print(f"Model v{version} → Staging")
    client.transition_model_version_stage(MODEL_NAME, version, "Production")
    print(f"Model v{version} → Production")


if __name__ == "__main__":
    main()
