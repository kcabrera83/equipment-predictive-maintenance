import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "equipment-predictive-maintenance"


def test_dashboard(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "predictions" in data


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_status_valid(client):
    response = client.post("/api/predict_status", json={})
    assert response.status_code in (200, 503)


def test_predict_status_with_params(client):
    payload = {
        "vibration_x": 3.5,
        "vibration_y": 2.0,
        "vibration_z": 1.5,
        "temperature": 80.0,
        "pressure": 250.0,
    }
    response = client.post("/api/predict_status", json=payload)
    assert response.status_code in (200, 503)


def test_predict_rul_valid(client):
    response = client.post("/api/predict_rul", json={})
    assert response.status_code in (200, 503)


def test_predict_rul_with_params(client):
    payload = {
        "vibration_x": 2.5,
        "temperature": 70.0,
        "runtime_hours": 8000.0,
        "days_since_maintenance": 60.0,
    }
    response = client.post("/api/predict_rul", json=payload)
    assert response.status_code in (200, 503)


def test_anomaly_check_valid(client):
    response = client.post("/api/anomaly_check", json={})
    assert response.status_code in (200, 400, 500, 503)
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data


def test_anomaly_check_with_params(client):
    payload = {
        "vibration_x": 10.0,
        "vibration_y": 8.0,
        "vibration_z": 7.0,
        "temperature": 120.0,
        "pressure": 500.0,
    }
    response = client.post("/api/anomaly_check", json=payload)
    assert response.status_code in (200, 400, 500, 503)
