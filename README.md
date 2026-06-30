## GRAiZE — Geo-satellite Remote AI analyZer for livestock hEalth

FastAPI service that receives sensor telemetry from livestock collars and persists it to a PostgreSQL database. Part of the AGRARIAN project, designed for smart livestock health monitoring in precision agriculture scenarios.
The application is designed for the livestock health monitoring use case within the AGRARIAN project. It was developed by STAMTech.

### Node Type
Edge / cloud — the service runs on any node reachable by the collar gateways and with access to the PostgreSQL instance.

### Application Objective
Expose an HTTP ingest endpoint that accepts JSON payloads from Raspberry Pi-based collar gateways, validates sensor readings (GNSS position, temperature, 3-axis acceleration, battery level), and writes each reading into three history tables: `Location_history`, `Feature_history`, and `Status_history`. Unknown device EUIs are automatically registered in `Devices_set` on first contact.

### Use Case Description
Livestock collars transmit periodic sensor data over LoRaWAN to a local Raspberry Pi gateway, which forwards the readings to this service via HTTP POST. The service acts as the ingestion layer between the field hardware and the analytical database, enabling downstream models to track animal location, activity patterns, and health indicators over time.

### Use instructions

**Environment variables** (supply via a `.env` file or shell environment):

| Variable | Description | Default |
|---|---|---|
| `DB_HOST` | PostgreSQL host | `10.160.101.177` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `graize_db` |
| `DB_USER` | Database user | *(required)* |
| `DB_PASSWORD` | Database password | *(required)* |

**Run the service:**

```bash
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 80
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest a sensor payload (returns 202) |

**Ingest payload example:**

```json
{
  "device_id": "AA:BB:CC:DD:EE:FF",
  "timestamp": "2026-06-25T09:15:00Z",
  "temperature": 38.6,
  "gnss": { "lat": 44.52, "lon": 8.91 },
  "acceleration": { "x": 0.12, "y": 0.42, "z": 0.08 },
  "battery": 84
}
```

**Output:** structured logs to stdout; rows inserted into `Location_history`, `Feature_history`, and `Status_history` in `graize_db`.

**Run tests:**

```bash
pytest src/
```