import pandas as pd
import requests
import time

df = pd.read_csv("data.csv")
speed_codes = df[df["VIOLATION_CODE"].str.contains("1180")]
coords = speed_codes[["Latitude", "Longitude"]]

highway_count = 0

for index, row in coords.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    
    query = f"""
    [out:json];
    way(around:10,{lat},{lon})[highway];
    out;
    """
    headers = {"User-Agent": "HighwayProject/1.0"}
    response = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, headers=headers)
    
    try:
        data = response.json()
    except:
        print("API error, skipping")
        time.sleep(1)
        continue

    elements = data["elements"]

    is_highway = False
    for element in elements:
        if element["tags"].get("highway") == "motorway":
            is_highway = True
            break
    
    if is_highway:
        highway_count += 1

    print(is_highway)
    time.sleep(1)

print(f"Highway coordinates: {highway_count}")