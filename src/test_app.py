from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "device_id": "cow_01",
    "timestamp": "2026-06-23T09:15:00Z",
    "temperature": 38.6,
    "gnss": {"lat": 44.52, "lon": 8.91},
    "acceleration": {"x": 0.12, "y": 0.42, "z": 0.08},
    "battery": 84,
}


def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "graize-app"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ingest_valid():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)  # simulate device already in DB
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("app.get_connection", return_value=mock_conn):
        r = client.post("/ingest", json=VALID_PAYLOAD)

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert body["device_id"] == "cow_01"
    assert "received_at" in body


def test_ingest_missing_field():
    r = client.post("/ingest", json={"device_id": "cow_01"})
    assert r.status_code == 422


def test_ingest_temperature_out_of_range():
    payload = {**VALID_PAYLOAD, "temperature": 50.0}
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422


def test_ingest_battery_out_of_range():
    payload = {**VALID_PAYLOAD, "battery": 110}
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422


def test_ingest_invalid_gnss():
    payload = {**VALID_PAYLOAD, "gnss": {"lat": 200.0, "lon": 8.91}}
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422