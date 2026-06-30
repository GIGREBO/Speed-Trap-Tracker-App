import random
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd 

roads_gdf = gpd.read_file("nyc_roads.geojson")
roads_gdf = roads_gdf[roads_gdf["tiger:county"].str.contains("NY", na=False)]

highway_violations_clean = pd.read_csv("violations.csv")

if "index_right" in highway_violations_clean.columns:
    highway_violations_clean = highway_violations_clean.drop(columns=["index_right"])
highway_violations_clean = gpd.GeoDataFrame(
    highway_violations_clean,
    geometry=gpd.points_from_xy(highway_violations_clean["Longitude"], highway_violations_clean["Latitude"]),
    crs="EPSG:4326"
).to_crs("EPSG:2263")

negative_points=[]


def random_point_on_roads(roads_gdf):
    for i in range(141000):
        road = roads_gdf.sample(1).geometry.iloc[0]
        fraction = random.random()
        point = road.interpolate(fraction, normalized=True)
        negative_points.append(point)
    return negative_points

random_point_on_roads(roads_gdf)

negatives_gdf = gpd.GeoDataFrame(geometry=negative_points)
negatives_gdf = negatives_gdf.set_crs(roads_gdf.crs)

buffer_distance = 100  # ~50 meters in feet, EPSG:2263

positives_buffered = highway_violations_clean.copy()
positives_buffered["geometry"] = positives_buffered.geometry.buffer(buffer_distance)

too_close = gpd.sjoin(negatives_gdf, positives_buffered, predicate="within")

clean_negatives = negatives_gdf[~negatives_gdf.index.isin(too_close.index)]

print("negatives_gdf shape:", negatives_gdf.shape)
print("too_close shape:", too_close.shape)
print("clean_negatives shape:", clean_negatives.shape)

roads_gdf_join = roads_gdf.copy()
if "index_right" in roads_gdf_join.columns:
    roads_gdf_join = roads_gdf_join.drop(columns=["index_right"])

clean_negatives = gpd.sjoin_nearest(clean_negatives, roads_gdf_join)
clean_negatives = clean_negatives[~clean_negatives.index.duplicated(keep="first")]


clean_negatives.to_csv("negatives.csv", index=False)

neg_df = pd.read_csv("negatives.csv")


print("clean_negatives shape:", clean_negatives.shape)
print("clean_negatives crs:", clean_negatives.crs)
print("roads_gdf_join crs:", roads_gdf_join.crs)
print("roads_gdf_join shape:", roads_gdf_join.shape)
