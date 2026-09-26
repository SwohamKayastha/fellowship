"""
Evidently AI drift monitoring for Telco Churn.

Steps:
  1. Load CSV, split 70% reference / 30% current
  2. Inject synthetic drift into current set
  3. Run Data Drift + Target Drift + custom metric
  4. Save HTML report → log to MLflow
"""

import pathlib

import mlflow
import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.metrics import ColumnDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from sklearn.preprocessing import LabelEncoder

DATA_PATH = pathlib.Path(__file__).parent / "data" / "telco_churn.csv"
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telco_churn_experiment"


def load_and_encode(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = le.fit_transform(df[col])
    return df


def inject_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()

    # Numeric drift: shift MonthlyCharges by up to ±20
    df["MonthlyCharges"] += rng.uniform(-20, 20, size=len(df))
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0)

    # Categorical drift: oversample Contract == Month-to-month (encoded as 0)
    month_to_month_mask = df["Contract"] == 0
    extra = df[month_to_month_mask].sample(frac=0.4, random_state=42, replace=True)
    df = pd.concat([df, extra], ignore_index=True)

    # Optional label drift: flip 5% of churn labels
    flip_idx = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[flip_idx, "Churn"] = 1 - df.loc[flip_idx, "Churn"]

    return df


def run_monitoring():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_encode(DATA_PATH)

    split = int(0.7 * len(df))
    reference = df.iloc[:split].reset_index(drop=True)
    current_raw = df.iloc[split:].reset_index(drop=True)

    rng = np.random.default_rng(seed=42)
    current = inject_drift(current_raw, rng)

    print(f"Reference size: {len(reference)} | Current size (post-drift): {len(current)}")
    print(f"Reference MonthlyCharges mean: {reference['MonthlyCharges'].mean():.2f}")
    print(f"Current  MonthlyCharges mean:  {current['MonthlyCharges'].mean():.2f}")

    column_mapping = ColumnMapping(
        target="Churn",
        numerical_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=["Contract", "InternetService", "PaymentMethod"],
    )

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        # Custom metric: drift specifically on MonthlyCharges
        ColumnDriftMetric(column_name="MonthlyCharges"),
    ])

    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    report_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(report_path))
    print(f"Report saved: {report_path}")

    # Interpret results
    result = report.as_dict()
    drift_metrics = result.get("metrics", [])
    print("\n--- Drift Summary ---")
    for m in drift_metrics:
        metric_id = m.get("metric", "")
        if "DatasetDriftMetric" in metric_id or "ColumnDriftMetric" in metric_id:
            val = m.get("result", {})
            print(f"  {metric_id}: {val}")

    # Log to MLflow under a dedicated monitoring run
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_artifact(str(report_path))
        mlflow.log_metric("reference_size", len(reference))
        mlflow.log_metric("current_size", len(current))
        mlflow.log_metric(
            "monthly_charges_mean_shift",
            abs(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean()),
        )
        print("Drift report logged to MLflow.")


if __name__ == "__main__":
    run_monitoring()
