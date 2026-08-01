import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import pickle
df = pd.read_csv("original_training.csv")

df["risk_tier"] = pd.cut(df["violation_count"], bins=[0, 5, 50, 500, 6542], labels=["low", "medium", "high", "very_high"], include_lowest=True)
grid_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"].apply(wkt.loads), crs="EPSG:2263")
grid_gdf = grid_gdf.to_crs("EPSG:4326")
df["longitude"] = grid_gdf.geometry.x
df["latitude"] = grid_gdf.geometry.y

df = df[["latitude", "longitude", "peak_hour", "peak_day", "peak_month", "violation_count", "risk_tier"]]


df.to_csv("training_data.csv", index=False)

X = df[["latitude", "longitude", "peak_hour", "peak_day", "peak_month"]]
y = df["risk_tier"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

# X_train["bridge"] = X_train["bridge"].fillna("no")
# X_test["bridge"] = X_test["bridge"].fillna("no")
# X_train["maxspeed"] = X_train["maxspeed"].fillna(X_train["maxspeed"].mode()[0])
# X_test["maxspeed"] = X_test["maxspeed"].fillna(X_train["maxspeed"].mode()[0])
# X_train["lanes"] = X_train["lanes"].fillna(X_train["lanes"].mode()[0])
# X_test["lanes"] = X_test["lanes"].fillna(X_train["lanes"].mode()[0])
# X_train["maxspeed"] = X_train["maxspeed"].str.replace(" mph", "").astype(int)
# X_test["maxspeed"] = X_test["maxspeed"].str.replace(" mph", "").astype(int)

# bridge_map = {"no": 0, "yes": 1, "movable": 2}
# oneway_map = {"yes": 1, "reversible": 0}

# X_train["bridge"] = X_train["bridge"].map(bridge_map)
# X_test["bridge"] = X_test["bridge"].map(bridge_map)

# X_train["oneway"] = X_train["oneway"].map(oneway_map)
# X_test["oneway"] = X_test["oneway"].map(oneway_map)

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


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")