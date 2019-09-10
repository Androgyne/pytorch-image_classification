import time
import datetime
import torch
from torch.nn import DataParallel
import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        self.model = DataParallel(network)

    def train(self, dataloaders, optimizer, criterion, scheduler, dataset_sizes):
        # Each epoch has a training and validation phase
        ret = {}
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   # Set model to evaluate mode

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
                    outputs = self.model(inputs)
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

            ret[phase] = epoch_acc
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        return ret

    def val(self, dataloaders, dataset_sizes):
        ret = {}
        for phase in ['train', 'val']:
            self.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            ret[phase] = epoch_acc
            print('{} Acc: {:.4f}'.format(
                phase, epoch_acc))


        ret['all'] = (ret['train'].item()*dataset_sizes['train'] + ret['val']*dataset_sizes['val'])/(dataset_sizes['val']+dataset_sizes['train'])
        return ret

