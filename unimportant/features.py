
df = pd.read_csv("violations.csv")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326")
roads_gdf = gpd.read_file("nyc_roads.geojson")
neg_df = pd.read_csv("negatives.csv")
neg_gdf = gpd.GeoDataFrame(neg_df, geometry=gpd.points_from_xy(neg_df["longitude"], neg_df["latitude"]), crs="EPSG:4326")

lat = 40.753044
lon = -73.775563

def extract_features(lat, lon):
    query = f"""
    [out:json];
    way(around:10,{lat},{lon})[highway];
    out;
    """
    headers = {"User-Agent": "HighwayProject/1.0"}
    response = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, headers=headers)
    data = response.json()
    elements = data["elements"]

    for element in elements:
        tags = element["tags"]
        if tags.get("highway") == "motorway":
            return {
                "lanes": tags.get("lanes"),
                "speed_limit": tags.get("maxspeed"),
                "bridge": tags.get("bridge"),
                "oneway": tags.get("oneway"),
                "highway": tags.get("highway")
            }
    
    return None

print(gdf)


print("geometry" in gdf.columns)
print(gdf.columns.tolist())


