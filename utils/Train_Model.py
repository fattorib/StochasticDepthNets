import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import uuid
import os
import wandb


# -------Benchmarks-------
# 10 epochs:
# 1070: 20.8 minutes ~ 16.15hrs to train model fully
# V100 w/ amp: 14 minutes ~11hrs to train model fully
# Switching to torch 1.6 gives the following speeds:
# 10 epochs:
# 1070: 12 minutes ~ Validating every second epoch: 8.3 hrs to train model fully
# P100: 6 minutes, fp32 ~ 6hr to train model fully - AMP did not do much to speed this up...


class Train_Model():

    def __init__(self, model, train_data, test_data, val_data):
        """General class for training and logging models

            - Logging is performed with Weights and Biases: <https://wandb.ai>

            - Learning rate scheduling/max_epochs are specific to training ResNet110


        """

        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

        # self.max_epochs = 500
        self.max_epochs = 200
        self.losses_increasing_stop = -1
        self.consecutive_losses_increasing = 0
        self.run_id = uuid.uuid4().hex

        if self.train_data is not None:
            self.trainloader = DataLoader(
                self.train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

            # Logging for Weights and Biases
            wandb.init(project="StochasticDepthResNets")
            wandb.run.name = self.run_id
            wandb.run.save()
            wandb.config.p_L = self.model.p_L
            wandb.config.num_layers = (6*self.model.N)+2
            wandb.config.max_epochs = self.max_epochs

        self.testloader = DataLoader(
            test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        if self.val_data is not None:
            self.valloader = DataLoader(
                val_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        self.val_losses = []
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.cuda()
        for e in range(0, self.max_epochs):
            running_loss = 0

            for images, labels in self.trainloader:

                images, labels = images.cuda(), labels.cuda()

                self.optimizer.zero_grad()

                output = self.model(images)

                loss = self.criterion(output, labels)
                running_loss += loss.item()

                loss.backward()

                self.optimizer.step()

            if (e+1) % 250 == 0 or (e+1) % 375 == 0:
                for g in self.optimizer.param_groups:
                    (g['lr']) = 0.1*(g['lr'])

            if e % 1 == 0:
                val_loss_epoch = self.validation_model(
                    self.valloader)
                wandb.log(
                    {'Validation Loss': val_loss_epoch, 'Training Loss': running_loss/len(self.trainloader)})

                self.val_losses.append(val_loss_epoch)
                if self.val_losses[-1] > min(self.val_losses):
                    self.consecutive_losses_increasing += 1
                else:
                    self.consecutive_losses_increasing = 0
                    self.save_model('best_weights')

            if e % 50 == 0:
                print(
                    f'Epoch:{e} - Train Loss: {running_loss/len(self.trainloader):.5f} - Validation Loss: {val_loss_epoch:.5f} - Consecutive Losses Increasing: {self.consecutive_losses_increasing}')

            if e % 50 == 0:
                self.eval(train=False)

            if self.consecutive_losses_increasing == self.losses_increasing_stop:
                print(f'Training ceased at {e} epochs.')
                self.save_model('last_weights')
                break

        self.save_model('last_weights')

    def eval(self, train):
        accuracy = 0

        if train:
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.trainloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    ps = F.softmax(self.model.forward(inputs), dim=1)
                    # Calculate accuracy
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            self.model.train()
            print(
                f'Train Accuracy: {100*accuracy/len(self.trainloader):.2f}%\n')
        else:
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    ps = F.softmax(self.model.forward(inputs), dim=1)
                    # Calculate accuracy
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            self.model.train()
            print(
                f'Test Accuracy: {100*accuracy/len(self.testloader):.2f}%\n')

    def validation_model(self, dataloader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.cuda(), labels.cuda()
                output = self.model(images)

                loss = self.criterion(output, labels)
                val_loss += loss.item()
        self.model.train()
        return val_loss/len(dataloader)

    def save_model(self, name):
        try:
            os.makedirs(f'StochasticDepthResNets/weights/{self.run_id}')
        except OSError:
            pass
        torch.save(self.model.state_dict(),
                   f'StochasticDepthResNets/weights/{self.run_id}/{name}.pth')
