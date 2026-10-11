"""Collect Falcon 9 records with SpaceX API v4; preserve a documented frozen fallback."""
from pathlib import Path
import json
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.spacexdata.com/v4"


def collect_live():
    session = requests.Session()
    cache = {}
    def get(path):
        if path not in cache:
            response = session.get(f"{API}/{path}", timeout=40)
            response.raise_for_status()
            cache[path] = response.json()
        return cache[path]
    records = []
    for launch in get("launches/past"):
        if len(launch.get("payloads", [])) != 1 or len(launch.get("cores", [])) != 1:
            continue
        rocket = get(f"rockets/{launch['rocket']}")
        if rocket["name"] != "Falcon 9" or launch["date_utc"][:10] > "2020-11-13":
            continue
        payload = get(f"payloads/{launch['payloads'][0]}")
        pad = get(f"launchpads/{launch['launchpad']}")
        flight_core = launch["cores"][0]
        if not flight_core.get("core"):
            continue
        core = get(f"cores/{flight_core['core']}")
        outcome = f"{flight_core.get('landing_success')} {flight_core.get('landing_type')}"
        records.append(dict(FlightNumber=launch["flight_number"], Date=launch["date_utc"][:10],
                            BoosterVersion=rocket["name"], PayloadMass=payload.get("mass_kg"),
                            Orbit=payload.get("orbit"), LaunchSite=pad["name"], Outcome=outcome,
                            Flights=flight_core.get("flight"), GridFins=flight_core.get("gridfins"),
                            Reused=flight_core.get("reused"), Legs=flight_core.get("legs"),
                            LandingPad=flight_core.get("landpad"), Block=core.get("block"),
                            ReusedCount=core.get("reuse_count"), Serial=core.get("serial"),
                            Longitude=pad["longitude"], Latitude=pad["latitude"]))
    return pd.DataFrame(records)


def collect_or_fallback():
    try:
        frame = collect_live()
        frame.to_csv(ROOT / "data" / "api_live.csv", index=False)
        status = {"mode": "live", "rows": len(frame), "source": API}
    except requests.RequestException as exc:
        frame = pd.read_csv(ROOT / "data" / "dataset_part_1.csv")
        status = {"mode": "official course snapshot", "rows": len(frame), "live_error": str(exc),
                  "source": "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv"}
    (ROOT / "data" / "api_collection_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return frame, status


if __name__ == "__main__":
    frame, status = collect_or_fallback()
    print(json.dumps(status, indent=2))
    print(frame.head().to_string(index=False))
