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
import sklearn
from sklearn.model_selection import train_test_split



class MultiLabelDataset(Dataset):
    def __init__(self):

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_test = None, None, None, None, None
        self.mode = 'train'


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
            image = image.reshape((3, 224, 224))
            if len(image.shape) == 2:
                copied_images = [image.copy() for i in range(3)]
                image = np.stack(copied_images, axis=-1)
            self.images.append(image)
        self.images = np.array(self.images)

        self.normalize()


    def normalize(self):
        self.images = self.images/255    

    def __len__(self):
        if self.mode == 'train':
            return self.x_train.shape[0]
        elif self.mode == 'val':
            return self.x_val.shape[0]
        elif self.mode == 'test':
            return self.x_test.shape[0]
       
    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = {'images': self.x_train[idx], 'labels': self.y_train[idx]}
        elif self.mode == 'val':
            sample = {'images': self.x_val[idx], 'labels': self.y_val[idx]}
        elif self.mode == 'test':
            sample = {'images': self.x_test[idx], 'labels': self.y_test[idx]}


        sample = {'images': self.images[idx], 'labels': self.labels[idx]}
        return sample

    def train_val_test_split(self): 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=0.33, random_state=42)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.33, random_state=22)

    def set_mode(self, x):
        self.mode = x
        return len(self)


d = MultiLabelDataset()
d.train_val_test_split()

print(d.set_mode("train"))
print(d.set_mode("val"))
print(d.set_mode("test"))


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




