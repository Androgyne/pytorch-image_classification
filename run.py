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
from util import imshow, preprocess_data
from engine import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_dataloader(args):

    if args.test:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
    else:
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.dataset, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, class_names, dataset_sizes

def make_model(classes, args):
    if args.network == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))
        model = model.to(device)
    elif args.network == 'vgg16':
        __import__('ipdb').set_trace()
        a = 1
    else:
        print("You can finish this by yourself")
    return model

def make_trainer(model, args):

    return Trainer(model)

def main(args):
    # load_data
    dataloader, classes, datasizes = make_dataloader(args)

    model = make_model(classes, args)

    trainer = make_trainer(model, args)
    if args.test:
        best_model = torch.load('best.pth')
        trainer.model.load_state_dict(best_model['net'])
        acc = trainer.val(dataloader, datasizes)
        print('all: {} val: {}'.format(acc['all'].item(), acc['val']))
    else:
        best_model_wts = copy.deepcopy(trainer.model.state_dict())
        best_acc = 0.0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for i in range(args.epochs):
            acc = trainer.train(dataloader, optimizer, criterion, scheduler, datasizes)
            if acc['val'].item() > best_acc:
                best_acc = acc['val'].item()
                best_model_wts = copy.deepcopy(trainer.model.state_dict())
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        trainer.model.load_state_dict(best_model_wts)
        torch.save({
            'net': trainer.model.state_dict()
            }, 'best.pth')

def prepare_data(args):
    '''
    you need to modify this dir path
    input dir denote the original data, which may be not divided
    we recommend that you need to write your own preprocess_data
    the target of preprocess data is to divide original data into train and val
    the test dir may be like the following
    data/
       data_dir/
           train/
              class1/
              class2/
              ...
              classn/
           val/
              class1/
              class2/
              ...
              classn/

    '''
    input_dir = args.input_dir
    output_dir = args.output_dir
    ratio = args.train_test_ratio
    __import__('ipdb').set_trace)
    preprocess_data(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='main', type=str)
    parser.add_argument("--dataset", default='data/mitochondria', type=str)
    parser.add_argument("--network", default='resnet18', type=str)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--input_dir", default='/home/haotongl/datasets/JPEG', type=str)
    parser.add_argument("--output_dir", default='data/mitochondria', type=str)
    parser.add_argument("--train_test_ratio", default=0.7, type=int)
    args = parser.parse_args()
    globals()[args.type](args)
