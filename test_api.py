import sys
sys.path.insert(0, '.')
from app import app

from fastapi.testclient import TestClient
client = TestClient(app)
tests = [
    ('GET', '/api/health', None),
    ('GET', '/api/dashboard', None),
    ('POST', '/api/predict_status', {
        'vibration_x': 2.0, 'vibration_y': 1.5, 'vibration_z': 1.0,
        'temperature': 65, 'pressure': 200, 'flow_rate': 150,
        'motor_current': 30, 'rpm': 1800, 'bearing_temp': 75,
        'oil_level': 80, 'seal_pressure': 50, 'suction_pressure': 100,
        'discharge_pressure': 200, 'power_consumption': 45,
        'runtime_hours': 5000, 'days_since_maintenance': 30,
    }),
    ('POST', '/api/predict_rul', {
        'vibration_x': 3.5, 'vibration_y': 2.8, 'vibration_z': 2.0,
        'temperature': 85, 'pressure': 180, 'flow_rate': 120,
        'motor_current': 45, 'rpm': 1800, 'bearing_temp': 95,
        'oil_level': 60, 'seal_pressure': 65, 'suction_pressure': 90,
        'discharge_pressure': 170, 'power_consumption': 60,
        'runtime_hours': 8000, 'days_since_maintenance': 90,
    }),
    ('POST', '/api/anomaly_check', {
        'vibration_x': 8.0, 'vibration_y': 6.5, 'vibration_z': 5.0,
        'temperature': 120, 'pressure': 100, 'motor_current': 90,
        'bearing_temp': 130, 'power_consumption': 95,
    }),
]
for method, ep, body in tests:
    r = client.post(ep, json=body) if method == 'POST' else client.get(ep)
    status = "OK" if r.status_code == 200 else "FAIL"
    result = r.json()
    print(f"{ep}: {r.status_code} {status} {result.get('status', '')}")
    assert r.status_code == 200, f"FAILED: {ep} returned {r.status_code}"
print("All 5 API tests passed!")
