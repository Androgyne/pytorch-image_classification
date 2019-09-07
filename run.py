from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import re
from util import imshow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
    	transforms.RandomVerticalFlip(),
    	transforms.RandomAffine(180, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/mitochondria'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    __import__('ipdb').set_trace()


    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({
        'net': model.state_dict()
        }, 'best.pth')
    return model

def main():

    # load_model
    __import__('ipdb').set_trace()
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model = train_model(model, criterion, optimizer, scheduler)

def prepare_data():
    input_dir = '/home/haotongl/datasets/JPEG'
    output_dir = 'data/mitochondria'
    classes = os.listdir(input_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'train'))
    os.mkdir(os.path.join(output_dir, 'val'))
    for cls in classes:
        os.mkdir(os.path.join(output_dir, 'train', cls))
        os.mkdir(os.path.join(output_dir, 'val', cls))
        jpegs = os.listdir(os.path.join(input_dir, cls))
        idxs = np.random.random((len(jpegs)))
        for i in range(len(jpegs)):
            jpgs = re.split('\(|\)' ,jpegs[i])
            if cls == 'normal':
                jpg = '1\ '+'\('+jpgs[1]+'\)'+jpgs[2]
            else:
                jpg = jpgs[0]+'\('+jpgs[1]+'\)'+jpgs[2]
            if idxs[i] > 0.7:
                os.system('cp {} {}'.format(os.path.join(input_dir, cls, jpg),
                        os.path.join(output_dir, 'val', cls, jpg)
                        ))
            else:
                os.system('cp {} {}'.format(os.path.join(input_dir, cls, jpg),
                        os.path.join(output_dir, 'train', cls, jpg)
                        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='main', type=str)
    args = parser.parse_args()
    globals()[args.type]()
