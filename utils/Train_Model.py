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


import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import uuid
import os
import wandb


class Train_Model():

    def __init__(self, model, train_data, test_data, val_data, meta_config, lr_scheduler=True):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.batch_size = int(meta_config['batch size'])
        self.learning_rate_annealing = meta_config['lr annealing']

        """Base class for training models
            meta_config (dict): Dictionrary containing meta hyperparam such as initial learning rate, batch size
        """

        if self.train_data is not None:
            self.trainloader = DataLoader(
                self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.testloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        if self.val_data is not None:
            self.valloader = DataLoader(
                val_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.max_epochs = meta_config['max epochs']
        self.losses_increasing_stop = 50
        self.consecutive_losses_increasing = 0
        self.val_losses = []

        self.accumulation_steps = meta_config['accumulation steps']

        if meta_config['Optimizer'] == 'SGD':

            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=meta_config['initial lr'],
                momentum=0.9, weight_decay=meta_config['weight decay'], nesterov=True)

        if meta_config['Optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=meta_config['initial lr'], weight_decay=meta_config['weight decay'])

        if meta_config['Optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=meta_config['initial lr'], weight_decay=meta_config['weight decay'])

        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=meta_config['initial lr'], max_lr=4 *
                meta_config['initial lr'], step_size_up=2000,
                mode='triangular',
                cycle_momentum=False)

        self.criterion = nn.CrossEntropyLoss()

        self.run_id = uuid.uuid4().hex

        # Logging for Weights and Biases
        self.wandb_run = wandb.init(project=meta_config['project name'])
        wandb.run.name = self.run_id
        # wandb.run.save()
        wandb.config.max_epochs = self.max_epochs
        wandb.config.batch_size = meta_config['batch size']
        wandb.config.optimizer = meta_config['Optimizer']
        wandb.config.initial_lr = meta_config['initial lr']
        wandb.config.weight_decay = meta_config['weight decay']
        wandb.config.accumulation_steps = meta_config['accumulation steps']

        try:
            wandb.config.p_L = model.p_L

        except ModuleAttributeError:
            pass

    def train(self):
        self.model.cuda()
        for e in range(0, self.max_epochs):
            running_loss = 0
            self.optimizer.zero_grad()
            for i, (images, labels) in enumerate(self.trainloader):

                images, labels = images.cuda(), labels.cuda()

                self.optimizer.zero_grad()

                output = self.model(images)

                loss = self.criterion(output, labels)
                running_loss += loss.item()

                loss /= self.accumulation_steps

                loss.backward()
                # self.optimizer.step()

                if (i+1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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

            # Monitor validation loss for lr annealing
            if self.consecutive_losses_increasing == self.losses_increasing_stop:

                if self.learning_rate_annealing:
                    # Clear consecutive loss history
                    self.consecutive_losses_increasing = 0
                    # wandb.log({'Annealed Epoch': e})
                    print(f'Annealing learning rate at epoch {e}')
                    for g in self.optimizer.param_groups:
                        (g['lr']) = 0.1*(g['lr'])

                else:
                    print(f'Training ceased at {e} epochs.')
                    self.save_model('last_weights')
                    break

        self.save_model('last_weights')
        wandb.run.finish()

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

    def filter_weight_visualizer(self, e, ch=0):
        with torch.no_grad():
            # Generate an image of first layer filters and log them
            filter = self.model.first_layer.weight.data
            n, c, w, h, = filter.shape

            if c != 3:
                filter = filter[:, ch, :, :].unsqueeze(dim=1)

            grid = utils.make_grid(
                filter, nrow=4, normalize=True, scale_each=True)

            image_filter = grid.permute(1, 2, 0).detach().numpy()
            images = wandb.Image(
                image_filter, caption=f"Visualization of first layer filter weights at epoch{e}")

            wandb.log({"Filter weights": images})

    def save_model(self, name):
        try:
            os.makedirs(
                f'StochasticDepthResNets/weights/imagenette/{self.run_id}')
        except OSError:
            pass
        torch.save(self.model.state_dict(),
                   f'StochasticDepthResNets/weights/imagenette/{self.run_id}/{name}.pth')
