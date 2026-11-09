from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def valid_payload():
    return {
        "asset_id": "MOTOR-TEST",
        "temperature_c": 75,
        "vibration_mm_s": 3.2,
        "pressure_bar": 6.5,
        "energy_kw": 42,
        "operating_hours": 12000,
        "work_orders_90d": 1,
        "days_since_maintenance": 45,
        "asset_criticality": 3
    }

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_contract():
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["failure_probability"] <= 1
    assert data["risk_level"] in {"BAJO", "MEDIO", "ALTO"}
    assert data["human_review_required"] is True

def test_extreme_but_valid_reading():
    payload = valid_payload()
    payload.update(temperature_c=150, vibration_mm_s=25, work_orders_90d=30,
                   days_since_maintenance=1000, asset_criticality=5)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_level"] in {"BAJO", "MEDIO", "ALTO"}

def test_invalid_criticality_rejected():
    payload = valid_payload()
    payload["asset_criticality"] = 8
    assert client.post("/predict", json=payload).status_code == 422

def test_negative_vibration_rejected():
    payload = valid_payload()
    payload["vibration_mm_s"] = -1
    assert client.post("/predict", json=payload).status_code == 422

def test_unknown_field_rejected():
    payload = valid_payload()
    payload["secret"] = "unexpected"
    assert client.post("/predict", json=payload).status_code == 422
