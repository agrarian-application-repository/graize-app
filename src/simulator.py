#!/usr/bin/env python3
"""
Raspberry Pi simulator for network architecture validation.

Usage:
  # Local (Phase 2)
  python src/simulator.py --host localhost --port 8080

  # AGRARIAN Gateway after VPN fix (Phase 6)
  python src/simulator.py --host 192.168.1.2 --port 80

  # Burst test
  python src/simulator.py --host localhost --port 8080 --count 20 --interval 0.5
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

try:
    import urllib.request as urlrequest
    import urllib.error as urlerror
except ImportError:
    raise SystemExit("Python 3 required")

BASE_LAT = 44.52
BASE_LON = 8.91


def make_payload(device_id: str, index: int) -> dict:
    angle = index * 0.1
    return {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(38.0 + random.uniform(-0.5, 1.5), 2),
        "gnss": {
            "lat": round(BASE_LAT + 0.001 * math.sin(angle), 6),
            "lon": round(BASE_LON + 0.001 * math.cos(angle), 6),
        },
        "acceleration": {
            "x": round(random.uniform(-0.5, 0.5), 3),
            "y": round(random.uniform(-0.5, 0.5), 3),
            "z": round(0.9 + random.uniform(-0.1, 0.1), 3),
        },
        "battery": max(0, 90 - index),
    }


def send(url: str, payload: dict, timeout: int) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urlerror.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urlerror.URLError as e:
        return 0, {"error": str(e.reason)}
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="GRAiZE Raspberry simulator")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cow_01")
    parser.add_argument("--count", type=int, default=5, help="Number of packets to send")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between packets")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    parser.add_argument("--malformed", action="store_true", help="Send one malformed payload first")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/ingest"
    print(f"Target: {url}")
    print(f"Device: {args.device}  |  Packets: {args.count}  |  Interval: {args.interval}s")
    print("-" * 60)

    ok = fail = 0

    if args.malformed:
        print("[0] Sending malformed payload (missing required fields)...")
        status, body = send(url, {"device_id": "bad"}, args.timeout)
        tag = "OK" if status == 422 else "UNEXPECTED"
        print(f"    [{tag}] HTTP {status} -> {body}")
        print()

    for i in range(args.count):
        payload = make_payload(args.device, i)
        t0 = time.monotonic()
        status, body = send(url, payload, args.timeout)
        elapsed = (time.monotonic() - t0) * 1000

        if status == 202:
            ok += 1
            print(f"[{i+1:>3}] OK       {elapsed:6.1f}ms  temp={payload['temperature']}°C  bat={payload['battery']}%")
        else:
            fail += 1
            print(f"[{i+1:>3}] FAIL     {elapsed:6.1f}ms  HTTP {status}  {body}")

        if i < args.count - 1:
            time.sleep(args.interval)

    print("-" * 60)
    total = ok + fail
    rate = ok / total * 100 if total else 0
    print(f"Result: {ok}/{total} delivered  ({rate:.1f}%)")
    if total > 0:
        print("PASS" if rate >= 95 else "FAIL (below 95% KPI threshold)")


if __name__ == "__main__":
    main()