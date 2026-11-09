from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .model import load_model, FEATURES
from .schemas import AssetReading

app = FastAPI(
    title="Bimbo Predictive Maintenance Prioritization",
    description="Prototipo académico con datos sintéticos. La IA recomienda; el técnico decide.",
    version="1.0.0"
)
model, _, training_metrics = load_model()
STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model/metrics")
def metrics():
    return training_metrics

@app.post("/predict")
def predict(reading: AssetReading):
    row = {feature: getattr(reading, feature) for feature in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    probability = float(model.predict_proba(X)[0, 1])
    risk = "ALTO" if probability >= 0.70 else "MEDIO" if probability >= 0.35 else "BAJO"
    # Explicación transparente basada en umbrales operativos didácticos.
    factors = []
    checks = [
        (reading.temperature_c >= 90, "Temperatura elevada (≥90 °C)"),
        (reading.vibration_mm_s >= 7.1, "Vibración elevada (≥7.1 mm/s)"),
        (reading.work_orders_90d >= 4, "Órdenes de trabajo frecuentes (≥4 en 90 días)"),
        (reading.days_since_maintenance >= 180, "Mantenimiento no reciente (≥180 días)"),
        (reading.asset_criticality >= 4, "Criticidad alta del activo (4–5)")
    ]
    factors = [label for condition, label in checks if condition]
    if not factors:
        factors = ["No se activaron los umbrales explicativos predefinidos"]
    priority = "P1 - revisión prioritaria" if risk == "ALTO" else (
        "P2 - programar revisión" if risk == "MEDIO" else "P3 - seguimiento"
    )
    return {
        "asset_id": reading.asset_id,
        "failure_probability": round(probability, 4),
        "risk_level": risk,
        "priority": priority,
        "contributing_signals": factors,
        "recommendation": "Revisar la alerta y validar físicamente el activo. Esta salida no genera órdenes automáticamente.",
        "human_review_required": True,
        "data_notice": "Predicción experimental basada en un modelo entrenado con datos sintéticos."
    }
