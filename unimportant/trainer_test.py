import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import pickle
from sklearn.dummy import DummyClassifier
df = pd.read_csv("original_training.csv")



df["risk_tier"] = pd.cut(df["violation_count"], bins=[0, 5, 50, 500, 6542], labels=["low", "medium", "high", "very_high"], include_lowest=True)
grid_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"].apply(wkt.loads), crs="EPSG:2263")
grid_gdf = grid_gdf.to_crs("EPSG:4326")
df["longitude"] = grid_gdf.geometry.x
df["latitude"] = grid_gdf.geometry.y

df = df[["latitude", "longitude", "peak_hour", "peak_day", "peak_month", "violation_count", "risk_tier"]]


df.to_csv("training_data.csv", index=False)


manhattan_bronx = [
    "FDR Drive", "Henry Hudson Parkway", "Harlem River Drive", "West Side Highway",
    "Trans-Manhattan Expressway", "Trans-Manhattan Expressway Upper Level", "Trans-Manhattan Expressway Lower Level",
    "Lincoln Tunnel", "Lincoln Tunnel Expressway", "Lincoln Tunnel Helix",
    "Holland Tunnel", "Holland Tunnel Rotary", "Brooklyn-Battery Tunnel", "Queens-Midtown Tunnel",
    "Cross Bronx Expressway", "Bruckner Expressway", "Major Deegan Expressway",
    "Throgs Neck Expressway", "Throgs Neck Bridge", "Mosholu Parkway",
    "Bronx River Parkway", "New England Thruway", "Hutchinson River Expressway",
    "Hutchinson River Parkway", "Bronx-Whitestone Bridge", "Christopher Columbus Highway"
]

train = df[df["name"].isin(manhattan_bronx)]
test = df[~df["name"].isin(manhattan_bronx)]


X_train = train[["latitude", "longitude", "peak_hour", "peak_day", "peak_month"]]
y_train = train["risk_tier"]

X_test = test[["latitude", "longitude", "peak_hour", "peak_day", "peak_month"]]
y_test = test["risk_tier"]


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred[:10])
print(accuracy_score(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high", "very_high"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "medium", "high", "very_high"])
disp.plot()
plt.show()

importances = model.feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp)

train_pred = model.predict(X_train)
print("Train accuracy:", accuracy_score(y_train, train_pred))
print("Test accuracy:", accuracy_score(y_test, y_pred))

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(dummy.score(X_test, y_test))
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")