# -*- coding: utf-8 -*-

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Transform
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Dataset
class celebADataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, split_path, transform=None, mode="train"):
        self.data = []
        self.labels = {}
        self.n_label = 0
        mode_dict = {"train": "0", "val": "1", "test": "2"}
        image_names = []

        # Image directory
        image_dirs = [dir for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir))]
        image_dirs.sort()

        # Load split data
        with open(split_path, "r") as f:
            splits = f.readlines()

        # Create Data
        for name in image_dirs:
            if name not in self.labels:
                self.labels[name] = self.n_label
                self.n_label += 1

            image_name = [os.path.join(name, n) for n in os.listdir(os.path.join(base_dir, name)) if n.endswith((".jpg", ".png"))]
            image_name.sort()
            image_names.extend(image_name)
            
        for file_name, mode_line in zip(image_names, splits):
            if mode_line.rstrip().split()[1] == mode_dict[mode]:
                self.data.append([os.path.join(base_dir, file_name), self.labels[file_name.split("/")[0]]])
        
        # Augmentation
        self.transforms = transform

    def __getitem__(self, idx):
        label = self.data[idx][1]
        
        file_path = self.data[idx][0]
        image = Image.open(file_path)
        if self.transforms is None:
            tf = transforms.ToTensor()
            image = tf(image)
            return image, label
        
        # Augmentation
        image = self.transforms(image)
        return image, label
    
    def __len__(self):
        return len(self.data)


# Load Data
# Data Link: https://drive.google.com/drive/folders/1_k6e2HEs7Y5BKM2S2t1gkkSUWTQ-YUj8?usp=drive_link
base_dir = "Celebrity Faces Dataset"
split_path = "split.txt"

train_data = celebADataset(base_dir, split_path, transform=data_transforms["train"], mode="train")
val_data = celebADataset(base_dir, split_path, transform=data_transforms["val"], mode="val")

trainloader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
valloader = DataLoader(val_data, batch_size=12, shuffle=False, pin_memory=True)
