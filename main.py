import torch
import torch.nn as nn #torch's neural networks functions
import torch.optim as optim #used to define our optimisers
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder
import timm #good for loading architecturs specific to image classification

import matplotlib.pyplot as plt #data visualisation
#Data exploration
import pandas as pd
import numpy as np

#1. Set up the Data Set

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    #2 methods needed for pytorch
    def __len__(self): #DataLoader needs to know how many examples we have in a dataset once we have created it
        return len(self.data)

    def __getitem__(self, idx): #Takes an index, returns one item
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

#2.
#Create Pytorch Model

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCardClassifier, self).__init__()

        #Define all the parts of the Model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        #Cutting off the final output so it outputs what we have in self.classififer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        #Typical output size of the efficientnet_b0 model
        enet_out_size = 1280
        # Make a classifier (making the model output num_classes rather than enet_out_size)
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        #Connect those parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


#3. The Training Loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SimpleCardClassifier(num_classes=53)
model.to(device)

# Loss Fucntion (the deviation from what the model should have spit out)
criterion = nn.CrossEntropyLoss()
opitmiser = optim.Adam(model.parameters(), lr=0.001)


train_folder = './train'
valid_folder = './valid'
test_folder = './test'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, 32, shuffle=True)
valid_loader = DataLoader(valid_dataset, 32, shuffle=False)
test_loader = DataLoader(test_dataset, 32, shuffle=False)

#Epoch = one run through the whole training dataset
num_epoch = 5
train_losses, val_losses = [], []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        opitmiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opitmiser.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(valid_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation Loss: {val_loss}")