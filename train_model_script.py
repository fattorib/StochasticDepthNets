import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from StochasticDepthResNets.src.ResidualBlocks import ResNet110
from StochasticDepthResNets.utils.Train_Model import Train_Model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import time


def main():
    cudnn.benchmark = True

    # Data augmentation as outlined in paper
    transform_train = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4, fill=0,
                              padding_mode='constant'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = CIFAR10(root='/CIFAR', train=True,
                         download=False, transform=transform_train)

    test_data = CIFAR10(root='/CIFAR', train=False,
                        download=False, transform=transform_test)

    # Getting a validation set
    train_data, val_data = torch.utils.data.random_split(train_data, [
        45000, 5000])

    # Training ResNet110
    model = ResNet110(pretrained=False)

    model_class = Train_Model(model, train_data, test_data, val_data)
    t0 = time.time()
    model_class.train()
    t1 = time.time()
    print(f'Total training time:{(t1-t0)/60.:2f} minutes.')
    model_class.eval(train=False)
    model_class.eval(train=True)


if __name__ == '__main__':
    main()
