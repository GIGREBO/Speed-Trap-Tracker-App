# import pandas as pd

# labels_path = r'C:\Users\tasne\Desktop\HighwayProject\classifdataset\mldata.csv'
# df = pd.read_csv(labels_path)



# shoulder_positives = (df["shoulder"] == 1).sum()
# shoulder_negatives = (df["shoulder"] == 0).sum()

# gore_positives = (df["gore_area"] == 1).sum()
# gore_negatives = (df["gore_area"] == 0).sum()

# print(f"Shoulder - Positives: {shoulder_positives}, Negatives: {shoulder_negatives}")
# print(f"Gore Area - Positives: {gore_positives}, Negatives: {gore_negatives}")




import os
import re

folder = r"C:\Users\tasne\Desktop\not_viable"
extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

files = os.listdir(folder)

# 1. Filter only streetview images
streetview_images = [
    f
    for f in files
    if f.lower().startswith("photo") and f.lower().endswith(extensions)
]

# 2. Sort them strictly by their current numeric suffix (0, 1, 2, ... 39)
streetview_images.sort(
    key=lambda f: int(re.search(r"\d+", f).group())
    if re.search(r"\d+", f)
    else 0
)

# 3. Rename in exact order starting at 449
for number, filename in enumerate(streetview_images, start=633):
    old_path = os.path.join(folder, filename)
    extension = os.path.splitext(filename)[1]

    new_name = f"photo{number}{extension}"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")

print(f"Done! Renamed {len(streetview_images)} images starting from photo449.")

# import pandas as pd

# df = pd.read_csv(r"training_data_copy.csv")
# from math import radians, sin, cos, sqrt, atan2

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # meters
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat, dlon = lat2 - lat1, lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     return 2 * R * atan2(sqrt(a), sqrt(1-a))

# dists = [haversine(df.latitude[i], df.longitude[i], df.latitude[i+1], df.longitude[i+1])
#          for i in range(len(df) - 1)]

# import matplotlib.pyplot as plt
# plt.hist([d for d in dists if d < 500], bins=100)
# plt.xlabel("Distance to next row (meters, zoomed in)")
# plt.show()



# print(df['heading'].describe())
# df[['latitude', 'longitude', 'heading']].head(10)