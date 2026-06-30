import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer
import folium

df = pd.read_csv("violations.csv")

roads_gdf = gpd.read_file("nyc_roads.geojson")



violations_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326").to_crs("EPSG:2263")

if "index_right" in violations_gdf.columns:
    violations_gdf = violations_gdf.drop(columns=["index_right"])

grid_points = []
grid_attrs = []

for idx, road in roads_gdf.iterrows():
    line = road.geometry
    road_length = line.length
    
    spacing = 100  
    num_points = max(1, int(road_length / spacing))
    
    for i in range(num_points):
        fraction = i / num_points
        point = line.interpolate(fraction, normalized=True)
        grid_points.append(point)
        grid_attrs.append(road.drop("geometry").to_dict())

grid_gdf = gpd.GeoDataFrame(grid_attrs, geometry=grid_points, crs=roads_gdf.crs)




buffer_distance = 200  # feet, about 60 meters

grid_buffered = grid_gdf.copy()
grid_buffered["geometry"] = grid_buffered.geometry.buffer(buffer_distance)

if "index_right" in grid_buffered.columns:
    grid_buffered = grid_buffered.drop(columns=["index_right"])

joined = gpd.sjoin(violations_gdf, grid_buffered, predicate="within")


violation_counts = joined.groupby("index_right").size().reset_index(name="violation_count")

grid_gdf["violation_count"] = grid_gdf.index.map(violation_counts.set_index("index_right")["violation_count"]).fillna(0).astype(int)

#print(grid_gdf["violation_count"].describe())

hotspot_idx = grid_gdf["violation_count"].idxmax()
hotspot = grid_gdf.loc[hotspot_idx]

transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326")
lat, lon = transformer.transform(1044991.9028656713, 218453.66439919377)
#print(lat, lon)




def count_violations_nearby(lat, lon, buffer_feet=200):
    point = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs("EPSG:2263")
    
    buffered = point.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_feet)
    
    if "index_right" in violations_gdf.columns:
        violations_gdf.drop(columns=["index_right"], inplace=True)
    
    nearby = gpd.sjoin(violations_gdf, buffered, predicate="within")
    return len(nearby)
#print(count_violations_nearby(40.72593900443235, -73.76915869421657))




def top_hotspots(n=10):
    top = grid_gdf.loc[grid_gdf.groupby("name")["violation_count"].idxmax()]
    top = top.nlargest(n, "violation_count")
    top_latlon = top.to_crs("EPSG:4326")
    
    for idx, row in top_latlon.iterrows():
        print(f"{row['name']}: {row.geometry.y:.6f}, {row.geometry.x:.6f} — {int(row['violation_count'])} violations")

def create_hotspot_map(center=[40.73, -73.87], zoom=11, output="hotspots.html"):
    coord_freq = violations_gdf.groupby(["Latitude", "Longitude"]).size().reset_index(name="frequency")
    
    m = folium.Map(location=center, zoom_start=zoom)
    
    for idx, row in coord_freq.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=max(2, row["frequency"] / 100),
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['frequency']} violations"
        ).add_to(m)
    
    m.save(output)
    print(f"Map saved to {output}")



df["VIOLATION_DATE"] = pd.to_datetime(df["VIOLATION_DATE"])
df["hour"] = pd.to_datetime(df["VIOLATION_TIME"], format="%H:%M:%S").dt.hour
df["day_of_week"] = df["VIOLATION_DATE"].dt.dayofweek
df["month"] = df["VIOLATION_DATE"].dt.month

joined["hour"] = joined.index.map(df["hour"])
joined["day_of_week"] = joined.index.map(df["day_of_week"])
joined["month"] = joined.index.map(df["month"])

peak_hour = joined.groupby("index_right")["hour"].agg(lambda x: x.mode()[0])
peak_day = joined.groupby("index_right")["day_of_week"].agg(lambda x: x.mode()[0])
peak_month = joined.groupby("index_right")["month"].agg(lambda x: x.mode()[0])

grid_gdf["peak_hour"] = grid_gdf.index.map(peak_hour).fillna(-1).astype(int)
grid_gdf["peak_day"] = grid_gdf.index.map(peak_day).fillna(-1).astype(int)
grid_gdf["peak_month"] = grid_gdf.index.map(peak_month).fillna(-1).astype(int)


cols_to_keep = ['bridge', 'hgv', 'lanes', 'maxspeed', 'oneway', 'geometry', 'name', 'violation_count', 'peak_hour', 'peak_day', 'peak_month']

grid_gdf = grid_gdf[cols_to_keep]
print(grid_gdf.shape)
print(grid_gdf.head())





grid_gdf.to_csv("training_data.csv", index=False)
print("Saved training_data.csv")


