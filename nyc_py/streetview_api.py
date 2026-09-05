"""
Fetches 2000 Street View images for randomly sampled grid points from
training_data_copy.csv and saves them to classifdataset/new_batch3.
No model, no predictions — image collection only, for future labeling.
 
Duplicate avoidance: maintains a persistent log of previously fetched
lat/lng pairs at classifdataset/used_grid_points.csv, so future runs of
this script skip points you've already pulled. On its first-ever run
(when that log doesn't exist yet) it tries to bootstrap the exclusion
set from your earlier fetch_log.csv / inference_results.csv files, so
this batch avoids re-fetching points from new_batch and
new_batch/inference_test_50. It can't recover points from batches whose
logs no longer exist, but it's a real improvement over sampling blind.
"""
 
import os
import time
import requests
import pandas as pd
 
# ---------------- CONFIG ----------------
API_KEY = "AIzaSyCqwmDyiDNOiZ2_ILMmx61qKIwqOxJgIc4"
CLASSIFDATASET_DIR = r"C:\Users\tasne\Desktop\HighwayProject\classifdataset"
OUTPUT_DIR = os.path.join(CLASSIFDATASET_DIR, "new_batch3")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
USED_LOG_PATH = os.path.join(CLASSIFDATASET_DIR, "used_grid_points.csv")
 
SAMPLE_N = 2000
 
# Known earlier logs to bootstrap the exclusion set from, if they exist
BOOTSTRAP_LOGS = [
    os.path.join(CLASSIFDATASET_DIR, "new_batch", "fetch_log.csv"),
    os.path.join(CLASSIFDATASET_DIR, "new_batch", "inference_test_50", "inference_results.csv"),
]
 
 
def load_used_points():
    used = set()
 
    if os.path.exists(USED_LOG_PATH):
        prior = pd.read_csv(USED_LOG_PATH)
        used.update(zip(prior["latitude"], prior["longitude"]))
        print(f"Loaded {len(prior)} previously-used points from {USED_LOG_PATH}")
    else:
        for path in BOOTSTRAP_LOGS:
            if os.path.exists(path):
                try:
                    log_df = pd.read_csv(path)
                    used.update(zip(log_df["latitude"], log_df["longitude"]))
                    print(f"Bootstrapped {len(log_df)} points from {path}")
                except Exception as e:
                    print(f"Could not read {path}: {e}")
 
    return used
 
 
def append_used_points(new_points_df):
    write_header = not os.path.exists(USED_LOG_PATH)
    new_points_df[["latitude", "longitude"]].to_csv(
        USED_LOG_PATH, mode="a", header=write_header, index=False
    )
 
 
def fetch_streetview_image(lat, lng, heading, pitch=-10, fov=90, size="640x640"):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "return_error_code": "true",
        "key": API_KEY,
    }
    return requests.get(url, params=params)
 
 
# ---------------- MAIN ----------------
df = pd.read_csv("training_data_copy.csv")
 
used_points = load_used_points()
if used_points:
    before = len(df)
    df = df[~df.apply(lambda r: (r["latitude"], r["longitude"]) in used_points, axis=1)]
    print(f"Excluded {before - len(df)} already-fetched points, {len(df)} remaining to sample from")
 
sample_df = df.sample(n=SAMPLE_N, random_state=43).reset_index(drop=True)
 
results = []
 
for i, row in sample_df.iterrows():
    response = fetch_streetview_image(row["latitude"], row["longitude"], row["heading"])
 
    if response.status_code == 200:
        filename = f"streetview_{i}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        results.append({
            "filename": filename,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "heading": row["heading"],
            "status": "saved",
        })
    else:
        results.append({
            "filename": None,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "heading": row["heading"],
            "status": f"failed_{response.status_code}",
        })
 
    if i % 50 == 0:
        print(f"Processed {i}/{len(sample_df)}")
    time.sleep(0.05)
 
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "fetch_log.csv"), index=False)
 
saved_count = (results_df["status"] == "saved").sum()
print(f"\nSaved {saved_count} images out of {len(sample_df)} attempted")
 
# record these as used so the next batch (e.g. new_batch4) skips them too
append_used_points(results_df[results_df["status"] == "saved"])
