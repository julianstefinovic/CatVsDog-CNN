from pickletools import uint8
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('tkagg')
import os
from os import listdir
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

def load_images(load_dir:str):

    images = []
    labels = []
    image_list = listdir(load_dir)
    image_list.sort()

    for im_dir in image_list:
        im = Image.open(load_dir + im_dir)
        im = np.array(im)
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        im = torch.tensor(np.moveaxis(im, -1, 0)).float()
        im = im[None, :]
        label = None
        if im_dir[0] == "c":
            label = 0
        else:
            label = 1
        labels.append(label)
        images.append(im)

    return images, labels

class ImageDataset(Dataset):

    def __init__(self, input_arrays:list, targets:list):
        self.input_arrays = input_arrays
        self.targets = targets

    def __len__(self):
        return len(self.input_arrays)

    def __getitem__(self, index):

        #print(self.input_arrays.shape)
        return self.input_arrays[index], self.targets[index]

    def showitem(self, index):
        sample = dataset.__getitem__(index)[0]
        sample = np.swapaxes(sample, 1, 3)
        sample = np.swapaxes(sample, 1, 2)
        sample = sample[0].numpy().astype(int)
        plt.title("Sample: " + str(index))
        plt.imshow(sample)
        plt.show()

class BasicCNN(nn.Module):

    def __init__(self):

        super().__init__()
        #3x64x64
        self.c1 = nn.Conv2d(3, 10, 3) #10x62x62 
        self.p1 = nn.MaxPool2d((2,2)) #10x31x31
        self.c2 = nn.Conv2d(10, 20, 4) #20x28x28
        self.p2 = nn.MaxPool2d((2,2)) #20x14x14
        self.L1 = nn.Linear(3920, 100) #100
        self.L2 = nn.Linear(100, 20) #20
        self.L3 = nn.Linear(20, 1) #1

    def forward(self, x):

        x = torch.relu(self.c1(x))
        x = self.p1(x)
        x = torch.relu(self.c2(x))
        x = self.p2(x)
        x = torch.flatten(x)
        x = torch.tanh(self.L1(x))
        x = torch.tanh(self.L2(x))
        x = torch.tanh(self.L3(x))
        x = torch.sigmoid(x)

        return x


def train(train_loader:DataLoader, model:BasicCNN, optimizer=torch.optim.Adadelta, num_epochs=50, lr=1e-2):
    
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optimizer(model.parameters(), lr=lr) #Optimizer

    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        for x, target in train_loader:
            
            target = target.to(torch.float32)
            #x = x.to(device)
            output = model(x[0])

            loss = loss_function(output, target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.detach().cpu().numpy())
        print(np.mean(epoch_losses))
        
        #if(save_model):
        #    model.save(filename=saved_model_filename)

    return model

train_images, train_labels = load_images(load_dir="dogs-vs-cats/train_subset/")
dataset = ImageDataset(train_images, train_labels)

train_size = int(len(train_images) / 2)
test_size = int(len(train_images) / 2)
#val_size = int(len(train_images) / 4)

print(train_size, test_size)
print(dataset.__len__())

train_set, test_set = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

model = BasicCNN()

model = train(train_dataloader, model)

sample = dataset.__getitem__(30)[0]
print(model.forward(sample))

sample = dataset.__getitem__(180)[0]
print(model.forward(sample))

sample = dataset.__getitem__(14)[0]
print(model.forward(sample))

sample = dataset.__getitem__(156)[0]
print(model.forward(sample))

sample = dataset.__getitem__(96)[0]
print(model.forward(sample))

sample = dataset.__getitem__(175)[0]
print(model.forward(sample))

dataset.showitem(50)