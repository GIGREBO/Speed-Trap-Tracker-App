import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import cv2 
import torch.nn as nn 
from torchvision import models
from torchvision.models import ResNet18_Weights


class MultiLabelDataset(Dataset):
    def __init__(self):
        labels_path = r'C:\Users\tasne\Desktop\HighwayProject\classifdataset\mldata.csv'
        images_path = r'C:\Users\tasne\Desktop\HighwayProject\classifdataset\projectpics'

        df = pd.read_csv(labels_path)

        self.labels = []
        for index, row in df.iterrows():
            self.labels.append(list(row.iloc[1:]))
        self.labels = np.array(self.labels)

        self.images = []
        image_names = df['image'].tolist()
        for img_name in image_names:
            image_path = os.path.join(images_path, img_name + '.png')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = np.array(image)
            if len(image.shape) == 2:
                copied_images = [image.copy() for i in range(3)]
                image = np.stack(copied_images, axis=-1)
            self.images.append(image)
        self.images = np.array(self.images)

        self.normalize()


    def normalize(self):
        self.image = self.images/255    

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        sample = {'images': self.images[idx], 'labels': self.labels[idx]}
        return sample






d = MultiLabelDataset()

dataloader = DataLoader(d, shuffle = True, batch_size = 128, drop_last = False) 

for i, s in enumerate(dataloader):
    img = s['images'].squeeze()
    l = s['labels'].tolist()[0]
    # print(img.shape)
    # break
    plt.imshow(img)
    plt.show()
    if i > 5: 
        break


model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x 




# # how many images
# print(len(d))

# item = d[90]
# image = item['images']
# label = item['labels']
# print(image.shape)
# print(label)

# plt.imshow(image)
# plt.title(f"Labels: {label}")
# plt.show()