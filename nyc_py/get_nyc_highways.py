import requests
import json
import geopandas as gpd
from shapely.geometry import LineString 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
df = pd.read_csv("data.csv")



query = """
[out:json];
way["highway"="motorway"](40.49,-74.05,40.9,-73.71);
out geom;
"""

headers = {"User-Agent": "HighwayProject/1.0"}
response = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, headers=headers)
data = response.json()

with open("nyc_highways.json", "w") as f:
    json.dump(data, f)


roads = []
tags_list = []

for element in data["elements"]:
    if "geometry" in element and "tags" in element:
        coords = [(point["lon"], point["lat"]) for point in element["geometry"]]
        line = LineString(coords)
        roads.append(line)
        tags_list.append(element["tags"])

roads_gdf = gpd.GeoDataFrame(tags_list, geometry=roads)
roads_gdf = roads_gdf.set_crs("EPSG:4326").to_crs("EPSG:2263")
roads_gdf = roads_gdf[~roads_gdf["tiger:county"].str.contains("NJ", na=False)]
roads_gdf.to_file("nyc_roads.geojson", driver="GeoJSON")


roads_gdf.plot()
plt.show()

#print(roads_gdf.shape)

violations_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]))
violations_gdf = violations_gdf.set_crs("EPSG:4326").to_crs("EPSG:2263")

#print(violations_gdf.shape)

buffer_distance = 32.8  # 10 meters in feet, since EPSG:2263 uses feet

roads_buffered = roads_gdf.copy()
roads_buffered["geometry"] = roads_buffered.geometry.buffer(buffer_distance)

highway_violations = gpd.sjoin(violations_gdf, roads_buffered, predicate="within")

#print(highway_violations.shape)


#print(highway_violations.index.duplicated().sum())


highway_violations_clean = highway_violations[~highway_violations.index.duplicated(keep="first")]
#print(highway_violations_clean.shape)





violations_df = pd.DataFrame(highway_violations_clean)
violations_df.to_csv("violations.csv", index=False)



