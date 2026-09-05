"""
Batch inference script — runs the trained CV model on Street View images
fetched for grid points in training_data_copy.csv, and saves predictions
(not just images) to a results CSV.

Matches dataset_work.py exactly:
- Model: resnet18 backbone, fc = Sequential(Dropout(0.3), Linear(in_features, 3))
- Single model, 3 outputs in this order: shoulder, gore_area, viable
- Preprocessing: crop top 20% / bottom 90% (HighwayROICrop) -> resize 224x224
  -> normalize with ImageNet mean/std. No mask_image (it's commented out
  in dataset_work.py, so it's not part of the trained model's expectations).
"""

import requests
import numpy as np
import cv2
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import math

# ---------------- CONFIG ----------------
API_KEY = "AIzaSyCqwmDyiDNOiZ2_ILMmx61qKIwqOxJgIc4"
OUTPUT_DIR = r"C:\Users\tasne\Desktop\HighwayProject\classifdataset\3new_batch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = r"C:\Users\tasne\Desktop\HighwayProject\cv_model.pth"  # from the new torch.save line in dataset_work.py
LABEL_NAMES = ["shoulder", "gore_area", "viable"]  # matches mldata.csv column order after "image"
NUM_CLASSES = len(LABEL_NAMES)

SAMPLE_N = 2000  # bump to 7000 after this test batch looks good

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
class MLC(nn.Module):
    def __init__(self, num_classes=3):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


model = MLC(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # disables dropout automatically, so it's a no-op at inference time

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

ROI_TOP_RATIO = 0.20
ROI_BOTTOM_RATIO = 0.90


def preprocess(image_rgb):
    """Mirrors dataset_work.py's eval_transform pipeline:
    HighwayROICrop -> resize 224x224 -> normalize."""
    h, w = image_rgb.shape[:2]
    top = int(h * ROI_TOP_RATIO)
    bottom = int(h * ROI_BOTTOM_RATIO)
    cropped = image_rgb[top:bottom, :, :]

    resized = cv2.resize(cropped, (224, 224))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    chw = normalized.transpose(2, 0, 1)
    tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor


def predict_image(image_rgb):
    tensor = preprocess(image_rgb)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs  # array of length NUM_CLASSES, order = LABEL_NAMES


# ---------------- STREET VIEW FETCH ----------------
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


# ---------------- MAIN LOOP ----------------
df = pd.read_csv("training_data_copy.csv")
sample_df = df.sample(n=SAMPLE_N, random_state=43).reset_index(drop=True)

results = []

for i, row in sample_df.iterrows():
    response = fetch_streetview_image(row["latitude"], row["longitude"], row["heading"])

    record = {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "heading": row["heading"],
    }

    if response.status_code == 200:
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # optional: save the image too, useful for spot-checking predictions
        filename = f"streetview_{i}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img_bgr)

        probs = predict_image(img_rgb)
        for name, p in zip(LABEL_NAMES, probs):
            record[f"{name}_prob"] = float(p)

        record["filename"] = filename
        record["status"] = "predicted"
    else:
        for name in LABEL_NAMES:
            record[f"{name}_prob"] = None
        record["filename"] = None
        record["status"] = f"failed_{response.status_code}"

    results.append(record)

    if i % 10 == 0:
        print(f"Processed {i}/{len(sample_df)}")
    time.sleep(0.05)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "inference_results.csv"), index=False)

success_count = (results_df["status"] == "predicted").sum()
print(f"\nPredicted {success_count} out of {len(sample_df)} points")
print(results_df[[f"{name}_prob" for name in LABEL_NAMES]].describe())
df = pd.read_csv(r"training_data_copy.csv")


# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # meters
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat, dlon = lat2 - lat1, lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     return 2 * R * atan2(sqrt(a), sqrt(1-a))

# def calculate_bearing(lat1, lon1, lat2, lon2):
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlon = lon2 - lon1
#     x = sin(dlon) * cos(lat2)
#     y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
#     bearing = degrees(atan2(x, y))
#     return (bearing + 360) % 360

# # Threshold in meters — adjust based on what your zoomed-in histogram showed.
# # If points are ~100ft (~30m) apart on the same road, something like 50m
# # is a reasonable cutoff between "same road" and "a jump to a new one."
# SAME_ROAD_THRESHOLD = 50

# headings = []
# n = len(df)

# for i in range(n):
#     if i < n - 1:
#         dist = haversine(df.latitude[i], df.longitude[i], df.latitude[i+1], df.longitude[i+1])
#         if dist < SAME_ROAD_THRESHOLD:
#             heading = calculate_bearing(df.latitude[i], df.longitude[i], df.latitude[i+1], df.longitude[i+1])
#             headings.append(heading)
#             continue
#     headings.append(headings[-1] if headings else 0)

# print("length of headings list:", len(headings))
# print("length of df:", len(df))

# df['heading'] = headings
# print("columns after assignment:", df.columns.tolist())


# df.to_csv(r"training_data_copy.csv", index=False)