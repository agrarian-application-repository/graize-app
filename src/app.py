#!/usr/bin/env python3
import logging
import struct
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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
# Pydantic models
# ---------------------------------------------------------------------------

class GnssModel(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class AccelerationModel(BaseModel):
    x: float
    y: float
    z: float


class SensorPayload(BaseModel):
    device_id: str                              # EUI/MAC string from the collar
    timestamp: datetime
    temperature: float = Field(..., ge=30.0, le=45.0)
    gnss: GnssModel
    acceleration: AccelerationModel
    battery: int = Field(..., ge=0, le=100)


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


def pack_acceleration(acc: AccelerationModel) -> bytes:
    """
    Pack x, y, z floats into a compact binary blob (3 × 4 bytes = 12 bytes).
    Uses little-endian IEEE 754 single-precision floats, matching typical
    accelerometer output formats.
    """
    return struct.pack("<fff", acc.x, acc.y, acc.z)


def insert_location(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Location_history."""
    # Convert float degrees to Int32 (× 10^7) as per schema
    lat_int = int(round(payload.gnss.lat * 1e7))
    lon_int = int(round(payload.gnss.lon * 1e7))
    temp_int = int(round(payload.temperature * 10))   # store as tenths of degree

    cursor.execute(
        """
        INSERT INTO Location_history
            (Device_id, Latitude, Longitude, Altitude, Temperature, Time)
        VALUES
            (%s, %s, %s, NULL, %s, %s)
        """,
        (device_id, lat_int, lon_int, temp_int, payload.timestamp)
    )


def insert_feature(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Feature_history (raw acceleration blob)."""
    blob = pack_acceleration(payload.acceleration)
    unix_ts = int(payload.timestamp.timestamp())        # Int64 Unix timestamp

    cursor.execute(
        """
        INSERT INTO Feature_history
            (Device_id, Blob_features, Time_unix, Time)
        VALUES
            (%s, %s, %s, %s)
        """,
        (device_id, psycopg2.Binary(blob), unix_ts, payload.timestamp)
    )


def insert_status(cursor, device_id: int, payload: SensorPayload):
    """Insert one row into Status_history (battery level as status flag)."""
    cursor.execute(
        """
        INSERT INTO Status_history
            (Device_id, Status_flag, RSSI, SF, Time)
        VALUES
            (%s, %s, NULL, NULL, %s)
        """,
        (device_id, payload.battery, payload.timestamp)
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


@app.post("/ingest-test", status_code=200)
async def ingest_test(request: Request):
    body = await request.json()
    logger.info("INGEST-TEST | %s", body)
    return {"received": body}


@app.post("/ingest", status_code=202)
async def ingest(payload: SensorPayload, request: Request):
    client = request.client.host if request.client else "unknown"
    logger.info(
        "INGEST | from=%s device=%s ts=%s temp=%.1f bat=%d%% gnss=(%.4f,%.4f) acc=(%.2f,%.2f,%.2f)",
        client,
        payload.device_id,
        payload.timestamp.isoformat(),
        payload.temperature,
        payload.battery,
        payload.gnss.lat,
        payload.gnss.lon,
        payload.acceleration.x,
        payload.acceleration.y,
        payload.acceleration.z,
    )

    try:
        conn = get_connection()
        with conn:                          # auto-commit on success, rollback on error
            with conn.cursor() as cur:
                # 1. Resolve device EUI → integer Device_id
                device_id = get_or_create_device(cur, payload.device_id)

                # 2. Insert into the three raw history tables
                insert_location(cur, device_id, payload)
                insert_feature(cur, device_id, payload)
                insert_status(cur, device_id, payload)

        conn.close()
        logger.info("DB inserts OK for device=%s", payload.device_id)

    except psycopg2.Error as e:
        logger.error("DB error: %s", e)
        raise HTTPException(status_code=500, detail="Database insert failed")

    return {
        "status": "accepted",
        "device_id": payload.device_id,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)