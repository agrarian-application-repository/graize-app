## GRAiZE — Geo-satellite Remote AI analyZer for livestock hEalth

FastAPI service that receives sensor telemetry from livestock collars and persists it to a PostgreSQL database. Part of the AGRARIAN project, designed for smart livestock health monitoring in precision agriculture scenarios.
The application is designed for the livestock health monitoring use case within the AGRARIAN project. It was developed by STAMTech.

### Node Type
Edge / cloud — the service runs on any node reachable by the collar gateways and with access to the PostgreSQL instance.

### Application Objective
Expose an HTTP ingest endpoint that accepts JSON payloads from Raspberry Pi-based collar gateways, validates sensor readings (GNSS position, temperature, status flags, and two accelerometer-derived feature blobs), and writes each reading into three raw history tables: `Location_history`, `Feature_history`, and `Status_history`. Unknown device EUIs are automatically registered in `Devices_set` on first contact.

On top of the raw ingest, every reading is run through two classifiers and the results are written into two activity tables:

- **Movement classification** — a sliding window of the device's last 12 GPS fixes (11 from `Location_history` + the one just arrived) is classified as `static` / `grazing` / `displaced` (see `src/classify_movement.py`) and logged as an interval in `Location_activities`.
- **Behaviour classification** — the two accelerometer feature blobs (`feature_1`, `feature_2`) are each classified independently with a trained model (`src/model_animal_150.pkl`, e.g. `laying_resting` / `ruminating` / `foraging`). Since the two predictions can disagree, each reading's time slice is split in half between them and written as two separate rows into `Inertial_activities` (no merging). Because the split needs the next reading's timestamp, a reading's own rows are only finalized once the *following* reading for that device arrives.

### Use Case Description
Livestock collars transmit periodic sensor data over LoRaWAN to a local Raspberry Pi gateway, which forwards the readings to this service via HTTP POST. The service acts as the ingestion and classification layer between the field hardware and the analytical database, enabling downstream models and dashboards to track animal location, movement, behaviour, and health indicators over time.

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
python src/app.py
```

Run from the repo root: `app.py`'s local imports (`classify_features`, `classify_movement`) expect `src/` to be the script's own directory, and `.env` is loaded relative to the working directory.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `GET` | `/health/db` | Database connectivity check |
| `POST` | `/ingest` | Ingest a sensor payload (returns 202) |
| `POST` | `/ingest-test` | Echoes back the received JSON body, no DB writes (returns 200) |

**Ingest payload example:**

```json
{
  "recv_time_utc": "2026-07-24T09:15:00Z",
  "dev_eui": "a8610a3432398609",
  "rssi": -70,
  "spreading_factor": 9,
  "status_flags": 1,
  "latitude": 443980648,
  "longitude": 89513693,
  "altitude": 63,
  "temperature": 29,
  "feature_1": "0181017a11ff",
  "feature_2": "017a017c0fff",
  "crc": 39345,
  "crc_ok": true
}
```

`latitude`/`longitude` are pre-scaled integers (decimal degrees × 1e7), matching how they're stored in `Location_history`. `feature_1`/`feature_2` are hex-encoded 6-byte accelerometer feature blobs.

**Ingest response example:**

```json
{
  "status": "accepted",
  "dev_eui": "a8610a3432398609",
  "received_at": "2026-07-24T09:15:00.123456+00:00",
  "classification": { "feature_1": "foraging", "feature_2": "foraging" },
  "movement": { "label": "grazing", "features": { "path_length_m": 42.1, "max_displacement_from_start_m": 18.3, "motion_fraction": 0.27, "n_fixes": 12 } }
}
```

**Output:** structured logs to stdout; rows inserted into `Location_history`, `Feature_history`, `Status_history`, `Location_activities`, and `Inertial_activities` in `graize_db`.

**Backfill:** `src/backfill_inertial_activities.py` recomputes behaviour classification for all existing `Feature_history` rows and repopulates `Inertial_activities` from scratch (clears the table first, then re-inserts). Run once from the repo root so `.env` resolves:

```bash
python src/backfill_inertial_activities.py
```

**Run tests:**

```bash
pytest src/
```