import numpy as np 
from torch.utils.data import Dataset 
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from PIL import Image
import os 


img_dir = r'C:\Users\tasne\Desktop\HighwayProject\classifdataset\projectpics'
path = r'C:\Users\tasne\Desktop\HighwayProject\classifdataset\mldata.csv'
df = pd.read_csv(path)


labels = []
for index, row in df.iterrows():
    labels.append(list(row.iloc[1:]))

labels = np.array(labels)

image_names = df['image'].tolist()

images = []
for img_name in image_names: 
    image_path = os.path.join(img_dir, img_name + '.png')
    print(image_path)
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    images.append(image)
images = np.array(images)


for idx, img in enumerate(images):
    plt.imshow(img)
    plt.show()
    if idx > 10: 
        break