import requests
import json


query = """
[out:json];
way(around:25, 40.61, -74.03)[highway];
out geom;
"""

headers = {"User-Agent": "HighwayProject/1.0"}
response = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, headers=headers)
data = response.json()

print(json.dumps(data, indent=4))


list = []

for key in data:
    for key in data["elements"]:
        list.append(key)


print(list)