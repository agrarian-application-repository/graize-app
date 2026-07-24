from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "recv_time_utc": "2026-07-02T15:11:29.624862+00:00",
    "dev_eui": "a8610a3432398609",
    "rssi": -93,
    "spreading_factor": 9,
    "status_flags": 1,
    "latitude": 443980648,
    "longitude": 89513693,
    "altitude": 63,
    "temperature": 29,
    "feature_1": "0181017a11ff",
    "feature_2": "017a017c0fff",
    "crc": 39345,
    "crc_ok": True,
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
    # fetchone() is called twice: get_or_create_device (device already in DB),
    # then get_last_feature_reading (no previous reading for this device).
    mock_cursor.fetchone.side_effect = [(1,), None]
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
    assert body["dev_eui"] == "a8610a3432398609"
    assert "received_at" in body


def test_ingest_missing_field():
    r = client.post("/ingest", json={"dev_eui": "a8610a3432398609"})
    assert r.status_code == 422


def test_ingest_missing_feature_blob():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "feature_2"}
    r = client.post("/ingest", json=payload)
    assert r.status_code == 422


def test_ingest_optional_fields_absent():
    payload = {k: v for k, v in VALID_PAYLOAD.items()
               if k not in ("rssi", "spreading_factor", "altitude", "temperature", "crc", "crc_ok")}

    mock_cursor = MagicMock()
    # fetchone() is called twice: get_or_create_device (device already in DB),
    # then get_last_feature_reading (no previous reading for this device).
    mock_cursor.fetchone.side_effect = [(1,), None]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("app.get_connection", return_value=mock_conn):
        r = client.post("/ingest", json=payload)

    assert r.status_code == 202