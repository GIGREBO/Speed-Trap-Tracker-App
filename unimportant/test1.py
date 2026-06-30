import pandas as pd
import requests 
import time
df = pd.read_csv("data.csv")

speed_codes = df[df["VIOLATION_CODE"].str.contains("1180D")]

coords = speed_codes[["Latitude","Longitude"]]


for index, row in coords.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    
    query = f"""
    [out:json];
    way(around:5,{lat},{lon})[highway];
    out;
    """
    headers = {"User-Agent": "HighwayProject/1.0"}
    response = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, headers=headers)
    try:
        data = response.json()
    except: 
        print("API error, skipping")
        continue
    time.sleep(1)


    elements = data["elements"]

    is_highway=False
    for element in elements:
        if element["tags"].get("highway") == "motorway":
            is_highway=True
            break
    print(is_highway)

  