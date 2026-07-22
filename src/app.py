#!/usr/bin/env python3
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from classify_features import load_model, dequantize
from classify_movement import extract_features, classify, LAT_LON_SCALE


# ---------------------------------------------------------------------------
# Configuration (set these as environment variables in your .env file)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    db_host: str = "10.160.101.177"
    db_port: int = 5432
    db_name: str = "graize_db"
    db_user: str | None = None
    db_password: str | None = None

settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("graize-app")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

# ---------------------------------------------------------------------------
# Classification (feature_1 / feature_2 → animal behaviour prediction)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_animal_150.pkl")

_model_bundle: dict | None = None


def get_model_bundle() -> dict:
    """Lazily load and cache the trained model + scaler (model_animal_150.pkl)."""
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = load_model(MODEL_PATH)
    return _model_bundle


def classify_feature_group(raw_bytes: bytes):
    """Dequantize one 6-byte uint8 feature group and classify it."""
    codes = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(1, -1)
    bundle = get_model_bundle()
    feats = dequantize(pd.DataFrame(codes))
    X = bundle["scaler"].transform(feats.values)
    pred = bundle["model"].predict(X)[0]
    return pred.item() if hasattr(pred, "item") else pred


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SensorPayload(BaseModel):
    recv_time_utc: datetime
    dev_eui: str                                 # EUI string from the collar
    rssi: int | None = None
    spreading_factor: int | None = None
    status_flags: int
    latitude: int                                # already scaled per Location_history schema
    longitude: int                               # already scaled per Location_history schema
    altitude: int | None = None
    temperature: int | None = None
    feature_1: str                               # hex-encoded bytes
    feature_2: str                               # hex-encoded bytes
    crc: int | None = None
    crc_ok: bool | None = None                    # informational only, not persisted


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_connection():
    """Open a new psycopg2 connection."""
    params = {
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
        "password": settings.db_password,
    }
    return psycopg2.connect(**params)


def get_or_create_device(cursor, eui: str) -> int:
    """
    Look up a device by its EUI. If it doesn't exist yet, insert it.
    Returns the integer Device_id.
    """
    cursor.execute(
        "SELECT Device_id FROM Devices_set WHERE EUI = %s",
        (eui,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Device not seen before — register it automatically
    cursor.execute(
        "INSERT INTO Devices_set (EUI) VALUES (%s) RETURNING Device_id",
        (eui,)
    )
    device_id = cursor.fetchone()[0]
    logger.info("Registered new device EUI=%s → Device_id=%d", eui, device_id)
    return device_id


def get_recent_locations(cursor, device_id: int, limit: int = 11):
    """Fetch the `limit` most recent (lat, lon, time) fixes for a device, oldest first."""
    cursor.execute(
        """
        SELECT Latitude, Longitude, Time
        FROM Location_history
        WHERE Device_id = %s
        ORDER BY Time DESC
        LIMIT %s
        """,
        (device_id, limit)
    )
    rows = cursor.fetchall()
    rows.reverse()
    return rows


def get_or_create_movement_activity(cursor, name: str) -> int:
    """
    Look up a movement activity by name. If it doesn't exist yet, insert it.
    Returns the integer Movement_activity_id.
    """
    cursor.execute(
        "SELECT Movement_activity_id FROM Movement_activities_set WHERE Name = %s",
        (name,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Label not seen before — register it automatically
    cursor.execute(
        "INSERT INTO Movement_activities_set (Name) VALUES (%s) RETURNING Movement_activity_id",
        (name,)
    )
    movement_activity_id = cursor.fetchone()[0]
    logger.info("Registered new movement activity name=%s → Movement_activity_id=%d",
                name, movement_activity_id)
    return movement_activity_id


def insert_location_activity(cursor, device_id: int, movement_activity_id: int,
                              time_start, time_end):
    """Insert one row into Location_activities."""
    cursor.execute(
        """
        INSERT INTO Location_activities
            (Device_id, Movement_activity_id, Time_start, Time_end)
        VALUES
            (%s, %s, %s, %s)
        """,
        (device_id, movement_activity_id, time_start, time_end)
    )


def insert_location(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Location_history."""
    cursor.execute(
        """
        INSERT INTO Location_history
            (Device_id, Latitude, Longitude, Altitude, Temperature, Time)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        """,
        (device_id, payload.latitude, payload.longitude, payload.altitude,
         payload.temperature, payload.recv_time_utc)
    )


def insert_feature(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Feature_history from the gateway's hex-encoded blobs."""
    feature_1 = bytes.fromhex(payload.feature_1)
    feature_2 = bytes.fromhex(payload.feature_2)
    unix_ts = int(payload.recv_time_utc.timestamp())    # Int64 Unix timestamp

    cursor.execute(
        """
        INSERT INTO Feature_history
            (Device_id, Feature_1, Feature_2, Time_unix, Time)
        VALUES
            (%s, %s, %s, %s, %s)
        """,
        (device_id, psycopg2.Binary(feature_1), psycopg2.Binary(feature_2),
         unix_ts, payload.recv_time_utc)
    )


def insert_status(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Status_history."""
    cursor.execute(
        """
        INSERT INTO Status_history
            (Device_id, Status_flag, RSSI, SF, Time)
        VALUES
            (%s, %s, %s, %s, %s)
        """,
        (device_id, payload.status_flags, payload.rssi, payload.spreading_factor,
         payload.recv_time_utc)
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def home():
    return {
        "name": "graize-app",
        "version": "1.0.0",
        "status": "running",
        "description": "GRAiZE - Geo-satellite Remote Ai analyZer for livestock hEalth",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/db")
async def health_db():
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "db_host": settings.db_host, "db_name": settings.db_name}
    except psycopg2.Error as e:
        logger.error("DB connection check failed: %s", e)
        raise HTTPException(status_code=503, detail="Database connection failed")


@app.post("/ingest-test", status_code=200)
async def ingest_test(request: Request):
    body = await request.json()
    logger.info("INGEST-TEST | %s", body)
    return {"received": body}


@app.post("/ingest", status_code=202)
async def ingest(payload: SensorPayload, request: Request):
    client = request.client.host if request.client else "unknown"
    logger.info(
        "INGEST | from=%s dev_eui=%s ts=%s rssi=%s sf=%s status=%s temp=%s gnss=(%s,%s) alt=%s",
        client,
        payload.dev_eui,
        payload.recv_time_utc.isoformat(),
        payload.rssi,
        payload.spreading_factor,
        payload.status_flags,
        payload.temperature,
        payload.latitude,
        payload.longitude,
        payload.altitude,
    )

    try:
        prediction_1 = classify_feature_group(bytes.fromhex(payload.feature_1))
        prediction_2 = classify_feature_group(bytes.fromhex(payload.feature_2))
        logger.info("CLASSIFY | dev_eui=%s pred_1=%s pred_2=%s",
                    payload.dev_eui, prediction_1, prediction_2)
    except Exception as e:
        logger.error("Classification failed for dev_eui=%s: %s", payload.dev_eui, e)
        prediction_1 = prediction_2 = None

    movement_features = None
    movement_label = None

    try:
        conn = get_connection()
        with conn:                          # auto-commit on success, rollback on error
            with conn.cursor() as cur:
                # 1. Resolve device EUI → integer Device_id
                device_id = get_or_create_device(cur, payload.dev_eui)

                # 2. Fetch the last 11 fixes (this one makes 12) for movement classification
                prior_fixes = get_recent_locations(cur, device_id, limit=11)

                # 3. Insert into the three raw history tables
                insert_location(cur, device_id, payload)
                insert_feature(cur, device_id, payload)
                insert_status(cur, device_id, payload)

                # 4. Classify movement over the sliding window and log the activity interval
                fixes = prior_fixes + [(payload.latitude, payload.longitude, payload.recv_time_utc)]
                lats = [lat / LAT_LON_SCALE for lat, lon, _ in fixes]
                lons = [lon / LAT_LON_SCALE for lat, lon, _ in fixes]

                try:
                    movement_features = extract_features(lats, lons)
                    movement_label = classify(movement_features)
                    movement_activity_id = get_or_create_movement_activity(cur, movement_label)
                    insert_location_activity(cur, device_id, movement_activity_id,
                                              time_start=fixes[0][2], time_end=fixes[-1][2])
                    logger.info("MOVEMENT | dev_eui=%s features=%s label=%s",
                                payload.dev_eui, movement_features, movement_label)
                except ValueError as e:
                    logger.info("MOVEMENT | dev_eui=%s skipped: %s", payload.dev_eui, e)

        conn.close()
        logger.info("DB inserts OK for dev_eui=%s", payload.dev_eui)

    except psycopg2.Error as e:
        logger.error("DB error: %s", e)
        raise HTTPException(status_code=500, detail="Database insert failed")

    return {
        "status": "accepted",
        "dev_eui": payload.dev_eui,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "classification": {
            "feature_1": prediction_1,
            "feature_2": prediction_2,
        },
        "movement": {
            "label": movement_label,
            "features": movement_features,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)