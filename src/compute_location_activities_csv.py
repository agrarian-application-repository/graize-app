"""
compute_location_activities_csv.py
─────────────────────────────────────────────────────────────────────────────
Read-only: replays the movement-classification logic over every row already
stored in Location_history and writes the resulting Location_activities rows
to a CSV file. Does not touch the database (no INSERT/UPDATE/DELETE).

Window rule per reading, replayed chronologically per device (matches the
/ingest logic in app.py):
    - Normal case: the last 11 valid prior fixes + this one (if this one is
      valid) = 12 fixes.
    - If this reading's own fix is invalid (out of GPS bounds, e.g. the
      INT32_MAX no-fix sentinel that decodes to lat=lon=214.7483647), it's
      excluded from the window; fall back to the last 12 valid fixes from
      history instead.
    - If fewer than 12 valid fixes are available either way, the reading is
      skipped (no activity row emitted for it).

Usage:
    python compute_location_activities_csv.py [output.csv]
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2

from app import settings, is_valid_coordinate
from classify_movement import extract_features, classify, LAT_LON_SCALE

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "location_activities.csv",
)


def compute_device_activities(device_id: int, readings):
    """
    readings: list of (Latitude, Longitude, Time) ordered chronologically for one device.
    Returns a list of (device_id, label, time_start, time_end) rows.
    """
    rows = []
    valid_so_far = []

    for latitude, longitude, time in readings:
        current_valid = is_valid_coordinate(latitude, longitude)
        window = valid_so_far[-11:] + [(latitude, longitude, time)] if current_valid \
            else valid_so_far[-12:]

        if len(window) == 12:
            lats = [lat / LAT_LON_SCALE for lat, lon, _ in window]
            lons = [lon / LAT_LON_SCALE for lat, lon, _ in window]
            label = classify(extract_features(lats, lons))
            rows.append((device_id, label, window[0][2], window[-1][2]))

        if current_valid:
            valid_so_far.append((latitude, longitude, time))

    return rows


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    conn = psycopg2.connect(**{
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
        "password": settings.db_password,
    })

    with conn.cursor() as cur:
        cur.execute("SELECT Device_id, EUI FROM Devices_set ORDER BY Device_id")
        eui_by_device = dict(cur.fetchall())

        cur.execute("SELECT DISTINCT Device_id FROM Location_history ORDER BY Device_id")
        device_ids = [r[0] for r in cur.fetchall()]

    all_rows = []
    for device_id in device_ids:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT Latitude, Longitude, Time
                FROM Location_history
                WHERE Device_id = %s
                ORDER BY Time
                """,
                (device_id,)
            )
            readings = cur.fetchall()

        device_rows = compute_device_activities(device_id, readings)
        all_rows.extend(device_rows)
        print(f"device_id={device_id} eui={eui_by_device.get(device_id)}: "
              f"{len(readings)} readings -> {len(device_rows)} activity rows")

    conn.close()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device_eui", "movement_classification", "time_start", "time_end"])
        for device_id, label, time_start, time_end in all_rows:
            writer.writerow([
                eui_by_device.get(device_id),
                label,
                time_start.isoformat(sep=" "),
                time_end.isoformat(sep=" "),
            ])

    print(f"\nDone. {len(all_rows)} rows written to {output_path}")


if __name__ == "__main__":
    main()
