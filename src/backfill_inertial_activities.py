"""
backfill_inertial_activities.py
─────────────────────────────────────────────────────────────────────────────
One-off backfill: recompute animal-behaviour predictions (prediction_1 /
prediction_2) for every row already stored in Feature_history, and populate
Inertial_activities with the results.

feature_1 and feature_2 are classified independently and can disagree, so
each reading's time slice [Time_i, Time_i+1) is split in half: the first
half is tagged with prediction_1, the second half with prediction_2. This
covers the full timeline with no gap or overlap between the two streams.
Each half-slice is inserted as its own Inertial_activities row — one row
per prediction, no merging of consecutive identical labels.

The last reading of a device has no following timestamp, so its slice has
zero duration and is skipped (Inertial_activities requires time_end >
time_start).

This script clears all existing Inertial_activities rows before
re-inserting, so it is safe to re-run.

Usage:
    python backfill_inertial_activities.py
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify_features import load_model, dequantize
from app import settings

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_animal_150.pkl")


def classify_feature_group(bundle: dict, raw_bytes: bytes):
    """Dequantize one 6-byte uint8 feature group and classify it."""
    codes = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(1, -1)
    feats = dequantize(pd.DataFrame(codes))
    X = bundle["scaler"].transform(feats.values)
    pred = bundle["model"].predict(X)[0]
    return pred.item() if hasattr(pred, "item") else pred


def get_or_create_inertial_activity(cursor, name: str) -> int:
    """
    Look up an inertial activity by name. If it doesn't exist yet, insert it.
    Returns the integer Inertial_activity_id.
    """
    cursor.execute(
        "SELECT Inertial_activity_id FROM Inertial_activities_set WHERE Name = %s",
        (name,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute(
        "INSERT INTO Inertial_activities_set (Name) VALUES (%s) RETURNING Inertial_activity_id",
        (name,)
    )
    inertial_activity_id = cursor.fetchone()[0]
    print(f"Registered new inertial activity name={name} -> Inertial_activity_id={inertial_activity_id}")
    return inertial_activity_id


def insert_inertial_activity(cursor, device_id: int, inertial_activity_id: int,
                              time_start, time_end):
    """Insert one row into Inertial_activities."""
    cursor.execute(
        """
        INSERT INTO Inertial_activities
            (Device_id, Inertial_activity_id, Time_start, Time_end)
        VALUES
            (%s, %s, %s, %s)
        """,
        (device_id, inertial_activity_id, time_start, time_end)
    )


def build_slots(readings):
    """
    readings: list of (time, pred_1, pred_2) ordered chronologically for one device.

    Splits each reading's [time_i, time_i+1) slice in half between pred_1
    (first half) and pred_2 (second half). Returns one (label, start, end)
    entry per prediction — no merging of consecutive identical labels.
    """
    slots = []
    n = len(readings)
    for i, (t, p1, p2) in enumerate(readings):
        t_next = readings[i + 1][0] if i + 1 < n else t
        mid = t + (t_next - t) / 2
        if t < mid:
            slots.append((p1, t, mid))
        if mid < t_next:
            slots.append((p2, mid, t_next))
    return slots


def main():
    conn = psycopg2.connect(
        host=settings.db_host, port=settings.db_port, dbname=settings.db_name,
        user=settings.db_user, password=settings.db_password,
    )
    bundle = load_model(MODEL_PATH)

    total_readings = 0
    total_slots = 0

    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM Inertial_activities")
            print(f"Cleared {cur.rowcount} existing Inertial_activities rows.")

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT Device_id FROM Feature_history ORDER BY Device_id")
        device_ids = [r[0] for r in cur.fetchall()]

    for device_id in device_ids:
        with conn:                          # one transaction per device
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT Time, Feature_1, Feature_2
                    FROM Feature_history
                    WHERE Device_id = %s
                    ORDER BY Time
                    """,
                    (device_id,)
                )
                rows = cur.fetchall()
                if not rows:
                    continue

                readings = [
                    (t, classify_feature_group(bundle, bytes(f1)),
                     classify_feature_group(bundle, bytes(f2)))
                    for t, f1, f2 in rows
                ]
                slots = build_slots(readings)

                for label, start, end in slots:
                    inertial_activity_id = get_or_create_inertial_activity(cur, label)
                    insert_inertial_activity(cur, device_id, inertial_activity_id, start, end)

                total_readings += len(rows)
                total_slots += len(slots)
                print(f"device_id={device_id}: {len(rows)} readings -> {len(slots)} rows")

    conn.close()
    print(f"\nDone. {total_readings} readings -> {total_slots} rows inserted into Inertial_activities.")


if __name__ == "__main__":
    main()
