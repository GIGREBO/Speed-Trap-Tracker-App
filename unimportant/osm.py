import math
import random
import time
import pandas as pd
import requests

NUM_COORDINATES = 300
MIN_DISTANCE_METERS = 1500
SEED = 42
random.seed(SEED)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

REGIONS = [
    {"name": "Northeast", "bbox": (38.0, -80.0, 43.5, -69.0)},
    {"name": "Southeast", "bbox": (25.0, -90.0, 38.0, -75.0)},
    {"name": "Midwest", "bbox": (36.0, -97.0, 47.5, -80.0)},
    {"name": "Texas", "bbox": (25.5, -106.5, 36.5, -93.5)},
    {"name": "California", "bbox": (32.0, -124.5, 42.0, -114.0)},
    {"name": "Pacific Northwest", "bbox": (42.0, -125.0, 49.0, -116.0)},
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def query_overpass_region(bbox):
    south, west, north, east = bbox

    query = f"""
    [out:json][timeout:120];
    (
      way["highway"="motorway"]["shoulder"~"^(no|none|0)$"]({south},{west},{north},{east});
      way["highway"="motorway"]["shoulder:both"~"^(no|none|0)$"]({south},{west},{north},{east});
      way["highway"="motorway"]["shoulder:width"="0"]({south},{west},{north},{east});

      way["highway"="motorway"]["bridge"]({south},{west},{north},{east});
      way["highway"="motorway"]["tunnel"]({south},{west},{north},{east});

      way["highway"="motorway"]["bridge"="viaduct"]({south},{west},{north},{east});
      way["highway"="motorway"]["embankment"]({south},{west},{north},{east});
      way["highway"="motorway"]["cutting"]({south},{west},{north},{east});

      way["highway"="motorway"]["layer"~"^[1-9]"]({south},{west},{north},{east});
      way["highway"="motorway"]["maxspeed"~"^(35|40|45|50|55)$"]({south},{west},{north},{east});
    );
    out tags geom;
    """

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": "HighwayShoulderCandidateResearch/5.0"},
                timeout=125,
            )

            if response.status_code == 200:
                return response.json().get("elements", [])

            print(f"  HTTP {response.status_code} from {endpoint}")

        except Exception as err:
            print(f"  Failed {endpoint}: {err}")

        time.sleep(2)

    return []

def calculate_score(tags):
    score = 0
    reasons = []

    shoulder = tags.get("shoulder", "").lower()
    shoulder_both = tags.get("shoulder:both", "").lower()
    shoulder_width = tags.get("shoulder:width", "").lower()

    if shoulder in {"no", "none", "0"}:
        score += 10
        reasons.append("explicit_no_shoulder")

    if shoulder_both in {"no", "none", "0"}:
        score += 10
        reasons.append("both_shoulders_absent")

    if shoulder_width == "0":
        score += 10
        reasons.append("zero_shoulder_width")

    if "tunnel" in tags:
        score += 9
        reasons.append("tunnel")

    if "bridge" in tags:
        score += 6
        reasons.append("bridge")

    if tags.get("bridge") == "viaduct":
        score += 3
        reasons.append("viaduct")

    if "cutting" in tags:
        score += 5
        reasons.append("cutting")

    if "embankment" in tags:
        score += 2
        reasons.append("embankment")

    try:
        layer = int(tags.get("layer", "0"))
        if layer > 0:
            score += min(layer * 2, 6)
            reasons.append("elevated")
    except ValueError:
        pass

    try:
        maxspeed = int(tags.get("maxspeed", "").replace(" mph", "").strip())

        if maxspeed <= 45:
            score += 5
            reasons.append("low_speed_motorway")
        elif maxspeed <= 55:
            score += 3
            reasons.append("moderate_speed_motorway")

    except ValueError:
        pass

    try:
        lanes = int(tags.get("lanes", "0"))

        if lanes >= 4:
            score += 2
            reasons.append("many_lanes")

        if lanes >= 5:
            score += 2

    except ValueError:
        pass

    return score, ", ".join(reasons)

all_candidates = []
seen_way_ids = set()

print("Searching for likely no-shoulder motorway locations...")

for region in REGIONS:
    print(f"\nQuerying {region['name']}...")

    ways = query_overpass_region(region["bbox"])

    print(f"Found {len(ways)} candidate motorway segments.")

    for way in ways:
        way_id = way.get("id")

        if way_id in seen_way_ids:
            continue

        seen_way_ids.add(way_id)

        geometry = way.get("geometry", [])
        tags = way.get("tags", {})

        if len(geometry) < 2:
            continue

        if tags.get("highway") != "motorway":
            continue

        score, reasons = calculate_score(tags)

        # Take the middle point of the OSM segment.
        midpoint = geometry[len(geometry) // 2]

        all_candidates.append({
            "latitude": midpoint["lat"],
            "longitude": midpoint["lon"],
            "score": score,
            "reason": reasons,
            "way_id": way_id,
            "road_name": tags.get("name", ""),
            "ref": tags.get("ref", ""),
            "lanes": tags.get("lanes", ""),
            "maxspeed": tags.get("maxspeed", ""),
            "bridge": tags.get("bridge", ""),
            "tunnel": tags.get("tunnel", ""),
            "layer": tags.get("layer", ""),
            "cutting": tags.get("cutting", ""),
            "embankment": tags.get("embankment", ""),
            "shoulder": tags.get("shoulder", ""),
            "shoulder_both": tags.get("shoulder:both", ""),
            "shoulder_left": tags.get("shoulder:left", ""),
            "shoulder_right": tags.get("shoulder:right", ""),
            "shoulder_width": tags.get("shoulder:width", ""),
        })

print(f"\nRaw candidate count: {len(all_candidates)}")

# Add small random noise so candidates with identical scores are mixed.
for candidate in all_candidates:
    candidate["_sort_score"] = candidate["score"] + random.random()

all_candidates.sort(
    key=lambda x: x["_sort_score"],
    reverse=True
)

selected = []

for candidate in all_candidates:
    lat = candidate["latitude"]
    lon = candidate["longitude"]

    if all(
        haversine(
            lat,
            lon,
            existing["latitude"],
            existing["longitude"]
        ) >= MIN_DISTANCE_METERS
        for existing in selected
    ):
        selected.append(candidate)

    if len(selected) >= NUM_COORDINATES:
        break

for candidate in selected:
    candidate.pop("_sort_score", None)

    lat = candidate["latitude"]
    lon = candidate["longitude"]

    candidate["google_street_view"] = (
        f"https://www.google.com/maps/@?api=1"
        f"&map_action=pano"
        f"&viewpoint={lat},{lon}"
    )

    candidate["google_maps"] = (
        f"https://www.google.com/maps/search/"
        f"?api=1&query={lat},{lon}"
    )

    candidate["openstreetmap"] = (
        f"https://www.openstreetmap.org/"
        f"?mlat={lat}&mlon={lon}"
        f"#map=19/{lat}/{lon}"
    )

df = pd.DataFrame(selected)

if not df.empty:
    df.insert(
        0,
        "candidate_number",
        range(1, len(df) + 1)
    )

output_file = "likely_no_shoulder_highways.csv"
df.to_csv(output_file, index=False)

print(f"\nSelected {len(df)} candidates.")
print(f"Saved to {output_file}")

print("\n" + "=" * 90)
print("TOP 20 CANDIDATES")
print("=" * 90)

for _, row in df.head(20).iterrows():
    route = row["ref"] or row["road_name"] or f"Way {row['way_id']}"

    print(
        f"#{int(row['candidate_number']):03d} | "
        f"Score {row['score']:02d} | "
        f"{route}"
    )

    print(
        f"  {row['latitude']:.6f}, "
        f"{row['longitude']:.6f}"
    )

    print(f"  Reason: {row['reason']}")
    print(f"  Street View: {row['google_street_view']}")