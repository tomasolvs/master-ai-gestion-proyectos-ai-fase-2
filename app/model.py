from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "maintenance_risk_model.joblib"
FEATURES = [
    "temperature_c", "vibration_mm_s", "pressure_bar", "energy_kw",
    "operating_hours", "work_orders_90d", "days_since_maintenance",
    "asset_criticality"
]

def make_synthetic_data(n=3000, seed=42):
    """Datos sintéticos reproducibles; no representan datos de Grupo Bimbo."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "temperature_c": rng.normal(72, 12, n).clip(35, 125),
        "vibration_mm_s": rng.gamma(2.2, 2.0, n).clip(0.1, 18),
        "pressure_bar": rng.normal(6.5, 1.0, n).clip(2, 11),
        "energy_kw": rng.normal(42, 9, n).clip(10, 85),
        "operating_hours": rng.integers(100, 30001, n),
        "work_orders_90d": rng.poisson(1.8, n).clip(0, 12),
        "days_since_maintenance": rng.integers(0, 731, n),
        "asset_criticality": rng.integers(1, 6, n)
    })
    # Etiqueta sintética: evento de falla en horizonte operativo simulado.
    score = (
        0.035 * (df.temperature_c - 70).clip(lower=0)
        + 0.48 * (df.vibration_mm_s - 3).clip(lower=0)
        + 0.10 * (df.pressure_bar - 7).abs()
        + 0.16 * df.work_orders_90d
        + 0.0020 * df.days_since_maintenance
        + 0.40 * (df.asset_criticality - 1)
        + 0.000018 * df.operating_hours
    )
    probability = 1 / (1 + np.exp(-(score - 5.3)))
    df["failure_within_horizon"] = rng.binomial(1, probability)
    return df

def train_model():
    data = make_synthetic_data()
    X, y = data[FEATURES], data["failure_within_horizon"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    return metrics

def load_model():
    if not MODEL_PATH.exists():
        train_model()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle.get("metrics", {})
